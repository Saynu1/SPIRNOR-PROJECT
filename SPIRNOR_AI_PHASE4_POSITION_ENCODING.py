#!/usr/bin/env python3
"""
SPIRNOR AI Phase 4: Deeper Architecture + SPIRNOR Position Encoding + Extended Generalization

Part A: Extended Generalization Depth
  Tests how far SPIRNOR's OOD advantage carries with a deeper (4-layer) network.
  - Tasks: GCD (pairwise), Smallest Prime Factor
  - Train 2-1000, test at 1K-2K, 2K-5K, 5K-10K, 10K-50K
  - Compare: Learned, SPIRNOR, Sinusoidal number embeddings

Part B: SPIRNOR Position Encoding (RoPE Replacement)
  Tests whether SPIRNOR-derived rotary frequencies outperform standard RoPE.
  Custom multi-head attention with true rotary applied to Q/K vectors.
  - Tasks requiring position understanding:
    1. Remainder: [a, b] -> a mod b (asymmetric, 2-token)
    2. Integer Division: [a, b] -> floor(a/b) (asymmetric, 2-token)
    3. Selective GCD: [a, b, c, d] -> GCD(a, d) ignoring b,c (4-token, attention routing)
  - Compare: No PE, Sinusoidal (additive), RoPE, SPIRNOR-RoPE, Random-RoPE
  - All use SPIRNOR number embedding (proven best from Phase 3)

Key insight: Standard RoPE uses geometric frequency progression (1/10000^(2d/D)),
which wastes dimensions on low frequencies for short sequences. SPIRNOR-RoPE uses
winding constants (pi, sqrt2, phi^2, e, ...) giving ALL dimensions meaningful
angular separation even for 2-4 token sequences.

Environment: Python 3.12, PyTorch 2.7.1+cpu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import json
import time
import sys
from collections import OrderedDict

# ============================================================
# CONSTANTS
# ============================================================

PHI = (1 + math.sqrt(5)) / 2

SPIRNOR_CONSTANTS = OrderedDict([
    ('pi', math.pi),
    ('sqrt2', math.sqrt(2)),
    ('phi_sq', PHI**2),
    ('e', math.e),
    ('golden_angle', 2 * math.pi / PHI**2),
    ('phi', PHI),
    ('ln2', math.log(2)),
    ('pi_e', math.pi / math.e),
])

SPIRNOR_CONST_LIST = list(SPIRNOR_CONSTANTS.values())

# ============================================================
# NUMBER EMBEDDINGS
# ============================================================

class SPIRNORNumberEmbedding(nn.Module):
    """Fixed SPIRNOR embedding — computes on the fly for any integer."""
    def __init__(self, d_model, constants=None):
        super().__init__()
        self.constants = list((constants or SPIRNOR_CONSTANTS).values())
        raw_dim = 7 * len(self.constants)  # 7 features per constant
        self.proj = nn.Linear(raw_dim, d_model)

    def _compute_features(self, nums):
        n_vals = nums.float()
        all_feats = []
        for C in self.constants:
            r = torch.log(torch.clamp(n_vals, min=1.0))
            theta = (C * n_vals) % (2 * math.pi)
            phi_angle = (PHI * n_vals) % (2 * math.pi)
            x = r * torch.sin(theta) * torch.cos(phi_angle)
            y = r * torch.sin(theta) * torch.sin(phi_angle)
            z = r * torch.cos(theta)
            feats = torch.stack([x, y, z,
                torch.sin(theta), torch.cos(theta),
                torch.sin(phi_angle), torch.cos(phi_angle)], dim=-1)
            all_feats.append(feats)
        return torch.cat(all_feats, dim=-1)

    def forward(self, nums):
        raw = self._compute_features(nums)
        return self.proj(raw)


class LearnedNumberEmbedding(nn.Module):
    """Standard learned embedding with modular hashing for OOD numbers."""
    def __init__(self, max_num, d_model):
        super().__init__()
        self.max_num = max_num
        self.embed = nn.Embedding(max_num + 1, d_model)

    def forward(self, nums):
        mapped = nums.long() % (self.max_num + 1)
        return self.embed(mapped)


class SinusoidalNumberEmbedding(nn.Module):
    """Sinusoidal embedding of numeric values (not positions)."""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, nums):
        n = nums.float().unsqueeze(-1)
        pe = torch.zeros(*nums.shape, self.d_model, device=nums.device)
        pe[..., 0::2] = torch.sin(n * self.div_term)
        pe[..., 1::2] = torch.cos(n * self.div_term)
        return self.proj(pe)


# ============================================================
# ROTARY POSITION ENCODINGS
# ============================================================

def apply_rotary(x, cos, sin):
    """Apply rotary embedding to x: (B, H, S, D)
    cos, sin: (1, 1, S, D//2) broadcast-ready.
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]
    d = min(x1.size(-1), cos.size(-1))
    out1 = x1[..., :d] * cos[..., :d] - x2[..., :d] * sin[..., :d]
    out2 = x1[..., :d] * sin[..., :d] + x2[..., :d] * cos[..., :d]
    result = torch.stack([out1, out2], dim=-1).flatten(-2)
    # If d_head is odd, append the remaining dimension unchanged
    if result.size(-1) < x.size(-1):
        result = torch.cat([result, x[..., result.size(-1):]], dim=-1)
    return result


class RotaryEmbedding(nn.Module):
    """Standard RoPE: θ_d = 1 / base^(2d/D), geometric frequency progression."""
    def __init__(self, d_head, max_len=512, base=10000):
        super().__init__()
        freqs = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        t = torch.arange(max_len, dtype=torch.float32)
        angles = torch.outer(t, freqs)
        self.register_buffer('cos_cached', angles.cos())
        self.register_buffer('sin_cached', angles.sin())

    def forward(self, x, seq_len):
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        return apply_rotary(x, cos, sin)


class SPIRNORRotaryEmbedding(nn.Module):
    """SPIRNOR-RoPE: uses winding constants as rotation frequencies.

    Instead of RoPE's geometric progression from base 10000, each dimension pair
    gets a frequency from a SPIRNOR winding constant. For d_head=16 with 8 constants,
    each pair maps to one constant:
      pair 0: freq = pi ≈ 3.14       pair 4: freq = golden_angle ≈ 2.40
      pair 1: freq = sqrt(2) ≈ 1.41  pair 5: freq = phi ≈ 1.62
      pair 2: freq = phi^2 ≈ 2.62    pair 6: freq = ln(2) ≈ 0.69
      pair 3: freq = e ≈ 2.72        pair 7: freq = pi/e ≈ 1.16

    All frequencies in [0.69, 3.14] → ALL dimension pairs provide meaningful
    angular separation even for 2-token sequences. Compare to standard RoPE
    where pairs 2-7 have freq < 0.03 and are nearly useless for short sequences.
    """
    def __init__(self, d_head, max_len=512, constants=None):
        super().__init__()
        consts = constants or SPIRNOR_CONST_LIST
        n_pairs = d_head // 2
        freqs = []
        for i in range(n_pairs):
            c_idx = i % len(consts)
            harmonic = (i // len(consts)) + 1
            freqs.append(consts[c_idx] * harmonic)
        freqs = torch.tensor(freqs, dtype=torch.float32)

        t = torch.arange(max_len, dtype=torch.float32)
        angles = torch.outer(t, freqs)
        self.register_buffer('cos_cached', angles.cos())
        self.register_buffer('sin_cached', angles.sin())

    def forward(self, x, seq_len):
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        return apply_rotary(x, cos, sin)


class RandomRotaryEmbedding(nn.Module):
    """Random frequency RoPE baseline — frequencies sampled uniformly [0.5, 5.0].
    Same range as SPIRNOR constants. If SPIRNOR-RoPE beats this, the advantage
    comes from the specific mathematical constants, not just the frequency range.
    """
    def __init__(self, d_head, max_len=512, seed=42):
        super().__init__()
        rng = np.random.RandomState(seed)
        freqs = torch.tensor(rng.uniform(0.5, 5.0, size=d_head // 2),
                             dtype=torch.float32)
        t = torch.arange(max_len, dtype=torch.float32)
        angles = torch.outer(t, freqs)
        self.register_buffer('cos_cached', angles.cos())
        self.register_buffer('sin_cached', angles.sin())

    def forward(self, x, seq_len):
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        return apply_rotary(x, cos, sin)


class SinusoidalPositionEncoding(nn.Module):
    """Additive sinusoidal PE (original Transformer)."""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)


# ============================================================
# CUSTOM TRANSFORMER WITH ROTARY PE SUPPORT
# ============================================================

class CustomMultiHeadAttention(nn.Module):
    """Multi-head attention with optional rotary PE applied to Q and K."""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

    def forward(self, x, rotary=None):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.nhead, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.nhead, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.nhead, self.d_head).transpose(1, 2)

        if rotary is not None:
            q = rotary(q, S)
            k = rotary(k, S)

        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(out)


class CustomTransformerLayer(nn.Module):
    """Pre-norm transformer layer with optional rotary PE."""
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.attn = CustomMultiHeadAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, rotary=None):
        x = x + self.attn(self.norm1(x), rotary=rotary)
        x = x + self.ff(self.norm2(x))
        return x


class FastTransformer(nn.Module):
    """Fast transformer using PyTorch's built-in TransformerEncoder.
    Used for Part A where no rotary PE is needed.
    """
    def __init__(self, d_model, nhead, num_layers, num_classes, num_embed):
        super().__init__()
        self.num_embed = num_embed
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, nums):
        x = self.num_embed(nums)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x)


class RotaryTransformer(nn.Module):
    """Transformer with custom attention supporting rotary PE.
    Used for Part B where we need true rotary comparisons.

    PE types:
      'none'         - no position encoding
      'sinusoidal'   - additive sinusoidal (original Transformer)
      'rope'         - true rotary (standard RoPE, base=10000)
      'spirnor_rope' - true rotary (SPIRNOR winding constant frequencies)
      'random_rope'  - true rotary (random frequencies, ablation baseline)
    """
    def __init__(self, d_model, nhead, num_layers, num_classes,
                 num_embed, pe_type='none'):
        super().__init__()
        self.num_embed = num_embed
        self.pe_type = pe_type
        d_head = d_model // nhead

        # Position encoding
        self.additive_pe = None
        self.rotary = None
        if pe_type == 'sinusoidal':
            self.additive_pe = SinusoidalPositionEncoding(d_model)
        elif pe_type == 'rope':
            self.rotary = RotaryEmbedding(d_head)
        elif pe_type == 'spirnor_rope':
            self.rotary = SPIRNORRotaryEmbedding(d_head)
        elif pe_type == 'random_rope':
            self.rotary = RandomRotaryEmbedding(d_head)

        self.layers = nn.ModuleList([
            CustomTransformerLayer(d_model, nhead, d_model * 4)
            for _ in range(num_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, nums):
        x = self.num_embed(nums)  # (B, seq_len, d_model)
        if self.additive_pe is not None:
            x = self.additive_pe(x)
        for layer in self.layers:
            x = layer(x, rotary=self.rotary)
        x = x.mean(dim=1)  # mean pool over sequence
        return self.head(x)


# ============================================================
# DATA GENERATION
# ============================================================

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def smallest_prime_factor(n):
    if n < 2:
        return 0
    if n % 2 == 0:
        return 2
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return i
    return n

GCD_BUCKETS = [1, 2, 3, 4, 5, 6, 10, 15]  # 9 classes
SPF_BUCKETS = [2, 3, 5, 7, 11, 13, 17, 19]  # 9 classes
REM_CLASSES = 8   # 0,1,2,3,4,5,6,7+
DIV_CLASSES = 7   # 0,1,2,3,4,5,6+
N_GCD_CLASSES = len(GCD_BUCKETS) + 1
N_SPF_CLASSES = len(SPF_BUCKETS) + 1

def gcd_to_class(g):
    for i, b in enumerate(GCD_BUCKETS):
        if g <= b:
            return i
    return len(GCD_BUCKETS)

def spf_to_class(spf):
    for i, b in enumerate(SPF_BUCKETS):
        if spf <= b:
            return i
    return len(SPF_BUCKETS)


def generate_gcd_data(n_samples, num_range, rng):
    lo, hi = num_range
    a = rng.randint(lo, hi + 1, size=n_samples)
    b = rng.randint(lo, hi + 1, size=n_samples)
    labels = np.array([gcd_to_class(gcd(int(a[i]), int(b[i])))
                       for i in range(n_samples)])
    inputs = np.stack([a, b], axis=1)
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def generate_spf_data(n_samples, num_range, rng):
    lo, hi = num_range
    nums = rng.randint(max(lo, 2), hi + 1, size=n_samples)
    labels = np.array([spf_to_class(smallest_prime_factor(int(n))) for n in nums])
    return (torch.tensor(nums, dtype=torch.long).unsqueeze(1),
            torch.tensor(labels, dtype=torch.long))


def generate_remainder_data(n_samples, num_range, rng):
    """[a, b] -> a mod b, classified into buckets. Position-dependent: a mod b != b mod a."""
    lo, hi = num_range
    a = rng.randint(lo, hi + 1, size=n_samples)
    b = rng.randint(max(lo, 2), hi + 1, size=n_samples)
    labels = np.array([min(int(a[i]) % int(b[i]), REM_CLASSES - 1)
                       for i in range(n_samples)])
    inputs = np.stack([a, b], axis=1)
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def generate_divfloor_data(n_samples, num_range, rng):
    """[a, b] -> floor(a/b), classified into buckets. Position-dependent."""
    lo, hi = num_range
    a = rng.randint(lo, hi + 1, size=n_samples)
    b = rng.randint(max(lo, 2), hi + 1, size=n_samples)
    labels = np.array([min(int(a[i]) // int(b[i]), DIV_CLASSES - 1)
                       for i in range(n_samples)])
    inputs = np.stack([a, b], axis=1)
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def generate_selective_gcd_data(n_samples, num_range, rng):
    """[a, b, c, d] -> GCD(a, d), ignoring b and c.
    Tests attention routing: model must learn positions 0 and 3 matter.
    4-token sequence — requires genuine position understanding.
    """
    lo, hi = num_range
    a = rng.randint(lo, hi + 1, size=n_samples)
    b = rng.randint(lo, hi + 1, size=n_samples)
    c = rng.randint(lo, hi + 1, size=n_samples)
    d = rng.randint(lo, hi + 1, size=n_samples)
    labels = np.array([gcd_to_class(gcd(int(a[i]), int(d[i])))
                       for i in range(n_samples)])
    inputs = np.stack([a, b, c, d], axis=1)
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


# ============================================================
# TRAINING
# ============================================================

def train_and_eval(model, train_inputs, train_labels, test_sets,
                   epochs=30, batch_size=256, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    n_params = sum(p.numel() for p in model.parameters())

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_inputs))
        total_loss = 0
        n_batches = 0
        for i in range(0, len(train_inputs), batch_size):
            idx = perm[i:i+batch_size]
            x, y = train_inputs[idx], train_labels[idx]
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        if (epoch + 1) in [1, 5, 10, 15, 20, 25, 30]:
            model.eval()
            with torch.no_grad():
                subset = min(5000, len(train_inputs))
                pred = model(train_inputs[:subset]).argmax(dim=-1)
                train_acc = (pred == train_labels[:subset]).float().mean().item()
            print(f"    Epoch {epoch+1:3d}/{epochs}: loss={total_loss/n_batches:.4f}, "
                  f"train_acc={train_acc:.4f}")

    train_time = time.time() - t0

    model.eval()
    results = OrderedDict()
    with torch.no_grad():
        for name, (test_in, test_lab) in test_sets.items():
            # Process in batches for large test sets
            all_preds = []
            for i in range(0, len(test_in), batch_size):
                batch = test_in[i:i+batch_size]
                preds = model(batch).argmax(dim=-1)
                all_preds.append(preds)
            all_preds = torch.cat(all_preds)
            acc = (all_preds == test_lab).float().mean().item()
            results[name] = acc

    return results, n_params, train_time


# ============================================================
# PART A: EXTENDED GENERALIZATION
# ============================================================

def run_part_a():
    print("=" * 70)
    print("PART A: EXTENDED GENERALIZATION DEPTH")
    print("=" * 70)
    print()
    print("Does SPIRNOR's OOD advantage persist with a deeper (4-layer) network")
    print("and extend to much larger number ranges (up to 50K)?")
    print()
    print("Architecture: 4-layer transformer, d_model=64, 4 heads, ~200K params")
    print("Train range: 2-1000 (2x Phase 3's range)")
    print()

    d_model = 64
    nhead = 4
    num_layers = 4
    epochs = 20
    rng = np.random.RandomState(42)

    train_range = (2, 1000)
    test_ranges = OrderedDict([
        ('in_range',    (2, 1000)),
        ('ood_1k_2k',   (1001, 2000)),
        ('ood_2k_5k',   (2001, 5000)),
        ('ood_5k_10k',  (5001, 10000)),
        ('ood_10k_50k', (10001, 50000)),
    ])

    all_results = {}

    for task_name, gen_fn, n_classes in [
        ('GCD', generate_gcd_data, N_GCD_CLASSES),
        ('Smallest Prime Factor', generate_spf_data, N_SPF_CLASSES),
    ]:
        print(f"\n{'='*60}")
        print(f"TASK: {task_name}")
        print(f"{'='*60}")
        print(f"  Train: {train_range}, Classes: {n_classes}")
        print(f"  Test ranges: {list(test_ranges.keys())}")

        train_in, train_lab = gen_fn(20000, train_range, rng)
        test_sets = OrderedDict()
        for name, r in test_ranges.items():
            ti, tl = gen_fn(3000, r, rng)
            test_sets[name] = (ti, tl)

        print(f"  Train: {train_in.shape}, label dist: {np.bincount(train_lab.numpy(), minlength=n_classes)}")

        task_results = OrderedDict()

        for embed_name in ['Learned', 'SPIRNOR', 'Sinusoidal']:
            print(f"\n  --- {embed_name} Embedding ---")

            if embed_name == 'Learned':
                num_embed = LearnedNumberEmbedding(train_range[1], d_model)
            elif embed_name == 'SPIRNOR':
                num_embed = SPIRNORNumberEmbedding(d_model)
            else:
                num_embed = SinusoidalNumberEmbedding(d_model)

            model = FastTransformer(
                d_model=d_model, nhead=nhead, num_layers=num_layers,
                num_classes=n_classes, num_embed=num_embed
            )

            results, n_params, train_time = train_and_eval(
                model, train_in, train_lab, test_sets,
                epochs=epochs, batch_size=512
            )

            task_results[embed_name] = {'params': n_params, 'time': train_time, **results}

            # Print results
            print(f"  {embed_name:12s} | {n_params:7d} params | ", end='')
            for name, acc in results.items():
                print(f"{name}={acc:.4f} ", end='')
            print(f"| {train_time:.1f}s")

        all_results[task_name] = task_results

    return all_results


# ============================================================
# PART B: POSITION ENCODING COMPARISON
# ============================================================

def run_part_b():
    print("\n" + "=" * 70)
    print("PART B: SPIRNOR POSITION ENCODING (RoPE REPLACEMENT)")
    print("=" * 70)
    print()
    print("Does SPIRNOR-derived rotary frequencies outperform standard RoPE?")
    print("True rotary applied to Q/K vectors in custom multi-head attention.")
    print()
    print("Key comparison:")
    print("  RoPE:         freq_d = 1/10000^(2d/D) — geometric, many low-freq dims wasted")
    print("  SPIRNOR-RoPE: freq_d = C_d (winding constant) — all dims meaningful")
    print("  Random-RoPE:  freq_d ~ Uniform[0.5, 5.0] — frequency range ablation")
    print()
    print("All use SPIRNOR number embedding (proven best from Phase 3).")
    print()

    d_model = 64
    nhead = 4
    num_layers = 3  # 3 layers for custom attention (faster on CPU)
    epochs = 20
    rng = np.random.RandomState(123)

    train_range = (2, 500)
    test_ranges = OrderedDict([
        ('in_range',    (2, 500)),
        ('ood_500_1k',  (501, 1000)),
        ('ood_1k_2k',   (1001, 2000)),
    ])

    pe_types = ['none', 'sinusoidal', 'rope', 'spirnor_rope', 'random_rope']
    pe_labels = {
        'none': 'No PE',
        'sinusoidal': 'Sinusoidal',
        'rope': 'RoPE',
        'spirnor_rope': 'SPIRNOR-RoPE',
        'random_rope': 'Random-RoPE',
    }

    all_results = OrderedDict()

    for task_name, gen_fn, n_classes in [
        ('Remainder',     generate_remainder_data,      REM_CLASSES),
        ('IntDiv',        generate_divfloor_data,        DIV_CLASSES),
        ('SelectiveGCD',  generate_selective_gcd_data,   N_GCD_CLASSES),
    ]:
        print(f"\n{'='*60}")
        print(f"TASK: {task_name} (position-dependent)")
        print(f"{'='*60}")
        print(f"  Train: {train_range}, Classes: {n_classes}")

        train_in, train_lab = gen_fn(15000, train_range, rng)
        test_sets = OrderedDict()
        for name, r in test_ranges.items():
            ti, tl = gen_fn(3000, r, rng)
            test_sets[name] = (ti, tl)

        seq_len = train_in.shape[1]
        print(f"  Sequence length: {seq_len}, Number embedding: SPIRNOR (fixed)")
        print(f"  Train: {train_in.shape}, label dist: {np.bincount(train_lab.numpy(), minlength=n_classes)}")

        task_results = OrderedDict()

        for pe_type in pe_types:
            label = pe_labels[pe_type]
            print(f"\n  --- {label} ---")

            num_embed = SPIRNORNumberEmbedding(d_model)
            model = RotaryTransformer(
                d_model=d_model, nhead=nhead, num_layers=num_layers,
                num_classes=n_classes, num_embed=num_embed, pe_type=pe_type
            )

            results, n_params, train_time = train_and_eval(
                model, train_in, train_lab, test_sets,
                epochs=epochs, batch_size=512
            )

            task_results[label] = {'params': n_params, 'time': train_time, **results}

            print(f"  {label:15s} | {n_params:7d} params | ", end='')
            for name, acc in results.items():
                print(f"{name}={acc:.4f} ", end='')
            print(f"| {train_time:.1f}s")

        all_results[task_name] = task_results

    return all_results


# ============================================================
# ANALYSIS AND SUMMARY
# ============================================================

def print_summary(results_a, results_b):
    print("\n" + "=" * 70)
    print("PHASE 4 RESULTS SUMMARY")
    print("=" * 70)

    # --- Part A ---
    print("\n" + "=" * 70)
    print("PART A: Extended Generalization")
    print("=" * 70)

    range_keys = ['in_range', 'ood_1k_2k', 'ood_2k_5k', 'ood_5k_10k', 'ood_10k_50k']
    range_labels = ['In-Range', '1K-2K', '2K-5K', '5K-10K', '10K-50K']

    for task, embeddings in results_a.items():
        print(f"\n  {task}:")
        header = f"  {'Embedding':12s} | {'Params':>7s}"
        for rl in range_labels:
            header += f" | {rl:>8s}"
        print(header)
        print(f"  {'-' * len(header)}")

        for embed_name, res in embeddings.items():
            line = f"  {embed_name:12s} | {res['params']:7d}"
            for key in range_keys:
                val = res.get(key, 0)
                best = max(e.get(key, 0) for e in embeddings.values())
                marker = '*' if abs(val - best) < 0.001 and val > 0 else ' '
                line += f" | {val:.4f}{marker}"
            print(line)

    # SPIRNOR generalization curve
    print("\n  SPIRNOR Generalization Curve (accuracy vs range):")
    for task, embeddings in results_a.items():
        spirnor = embeddings.get('SPIRNOR', {})
        print(f"    {task}:", end='')
        for key, label in zip(range_keys, range_labels):
            val = spirnor.get(key, 0)
            bar = '#' * int(val * 30)
            print(f"\n      {label:8s}: {val:.4f} |{bar}")
        print()

    # --- Part B ---
    print("\n" + "=" * 70)
    print("PART B: Position Encoding Comparison")
    print("=" * 70)

    range_keys_b = ['in_range', 'ood_500_1k', 'ood_1k_2k']
    range_labels_b = ['In-Range', '500-1K', '1K-2K']

    for task, pe_results in results_b.items():
        print(f"\n  {task}:")
        header = f"  {'PE Type':15s} | {'Params':>7s}"
        for rl in range_labels_b:
            header += f" | {rl:>8s}"
        print(header)
        print(f"  {'-' * len(header)}")

        for pe_name, res in pe_results.items():
            line = f"  {pe_name:15s} | {res['params']:7d}"
            for key in range_keys_b:
                val = res.get(key, 0)
                best = max(e.get(key, 0) for e in pe_results.values())
                marker = '*' if abs(val - best) < 0.001 and val > 0 else ' '
                line += f" | {val:.4f}{marker}"
            print(line)

    # --- Key comparisons ---
    print("\n" + "=" * 70)
    print("KEY COMPARISONS")
    print("=" * 70)

    # SPIRNOR-RoPE vs RoPE
    print("\n  SPIRNOR-RoPE vs RoPE (true rotary comparison):")
    for task, pe_results in results_b.items():
        rope = pe_results.get('RoPE', {})
        spirnor = pe_results.get('SPIRNOR-RoPE', {})
        print(f"\n    {task}:")
        for key, label in zip(range_keys_b, range_labels_b):
            r_val = rope.get(key, 0)
            s_val = spirnor.get(key, 0)
            diff = s_val - r_val
            if diff > 0.01:
                winner = "SPIRNOR-RoPE"
            elif diff < -0.01:
                winner = "RoPE"
            else:
                winner = "TIE"
            print(f"      {label:8s}: RoPE={r_val:.4f}  SPIRNOR-RoPE={s_val:.4f}  "
                  f"diff={diff:+.4f}  ({winner})")

    # SPIRNOR-RoPE vs Random-RoPE (does the specific constants matter?)
    print("\n  SPIRNOR-RoPE vs Random-RoPE (are the specific constants important?):")
    for task, pe_results in results_b.items():
        random = pe_results.get('Random-RoPE', {})
        spirnor = pe_results.get('SPIRNOR-RoPE', {})
        print(f"\n    {task}:")
        for key, label in zip(range_keys_b, range_labels_b):
            rr_val = random.get(key, 0)
            s_val = spirnor.get(key, 0)
            diff = s_val - rr_val
            if diff > 0.01:
                winner = "SPIRNOR constants matter"
            elif diff < -0.01:
                winner = "Random equally good"
            else:
                winner = "Similar"
            print(f"      {label:8s}: Random={rr_val:.4f}  SPIRNOR={s_val:.4f}  "
                  f"diff={diff:+.4f}  ({winner})")

    # No PE vs any PE (does position encoding help at all?)
    print("\n  No PE vs Best PE (is position encoding necessary?):")
    for task, pe_results in results_b.items():
        no_pe = pe_results.get('No PE', {})
        best_pe_name = None
        best_pe_val = -1
        for pe_name, res in pe_results.items():
            if pe_name == 'No PE':
                continue
            avg_ood = np.mean([res.get(k, 0) for k in range_keys_b[1:]])
            if avg_ood > best_pe_val:
                best_pe_val = avg_ood
                best_pe_name = pe_name
        best_pe = pe_results.get(best_pe_name, {})
        print(f"\n    {task} (best PE: {best_pe_name}):")
        for key, label in zip(range_keys_b, range_labels_b):
            n_val = no_pe.get(key, 0)
            b_val = best_pe.get(key, 0)
            diff = b_val - n_val
            print(f"      {label:8s}: No PE={n_val:.4f}  {best_pe_name}={b_val:.4f}  "
                  f"diff={diff:+.4f}")

    # Win counts
    print("\n" + "=" * 70)
    print("SCOREBOARD")
    print("=" * 70)

    pe_wins = {pe: 0 for pe in ['No PE', 'Sinusoidal', 'RoPE', 'SPIRNOR-RoPE', 'Random-RoPE']}
    total_comparisons = 0
    for task, pe_results in results_b.items():
        for key in range_keys_b:
            best_val = max(res.get(key, 0) for res in pe_results.values())
            for pe_name, res in pe_results.items():
                if abs(res.get(key, 0) - best_val) < 0.001:
                    pe_wins[pe_name] += 1
            total_comparisons += 1

    print(f"\n  Part B wins across {total_comparisons} comparisons "
          f"(3 tasks x {len(range_keys_b)} ranges):")
    for pe_name, wins in sorted(pe_wins.items(), key=lambda x: -x[1]):
        bar = '#' * (wins * 3)
        print(f"    {pe_name:15s}: {wins:2d}/{total_comparisons} wins  {bar}")


def main():
    print("=" * 70)
    print("SPIRNOR AI PHASE 4: DEEPER ARCHITECTURE + POSITION ENCODING")
    print("=" * 70)
    print()
    print("Two key questions:")
    print("  A) How far does SPIRNOR's OOD generalization extend with deeper networks?")
    print("  B) Can SPIRNOR-derived frequencies replace RoPE as position encoding?")
    print()

    results_a = run_part_a()
    results_b = run_part_b()

    print_summary(results_a, results_b)

    # Save results
    all_results = {
        'part_a': {k: dict(v) for k, v in results_a.items()},
        'part_b': {k: dict(v) for k, v in results_b.items()},
        'config': {
            'd_model': 64, 'nhead': 4, 'num_layers_a': 4, 'num_layers_b': 3,
            'train_range_a': [2, 1000], 'train_range_b': [2, 500],
            'spirnor_constants': dict(SPIRNOR_CONSTANTS),
        }
    }

    def clean(obj):
        if isinstance(obj, OrderedDict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    save_path = 'Scripts/SPIRNOR_AI_PHASE4_RESULTS.json'
    with open(save_path, 'w') as f:
        json.dump(clean(all_results), f, indent=2)
    print(f"\nResults saved to: {save_path}")


if __name__ == '__main__':
    main()
