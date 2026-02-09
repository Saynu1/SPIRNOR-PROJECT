#!/usr/bin/env python3
"""
SPIRNOR AI Phase 6: Token-Level Transformer with SPIRNOR Numeric Embedding

Building on the proven SPIRNOR advantage from Phases 3-4, this experiment
scales up the numeric embedding approach with:
  1. Transformer encoder (not MLP) for richer feature interaction
  2. Multi-task training across 5 number-theoretic tasks
  3. Large training data (250K examples) and eval sets (5000/range)
  4. Extended OOD ranges up to 250x training max
  5. Four configurations testing embedding type and position encoding

Key hypothesis: SPIRNOR's log-spiral mapping of integers encodes
divisibility and prime-factor structure that generalizes OOD because
the mapping is deterministic and structure-preserving for any integer.

Architecture:
  Input: [task_token, num1, num2_or_pad] (3 tokens)
  Encoder: 4-layer Transformer with bidirectional attention + RoPE
  Output: Task-specific 2-layer MLP classification heads

Configurations tested:
  1. Learned embedding + RoPE        (baseline)
  2. Sinusoidal embedding + RoPE     (fixed-feature baseline)
  3. SPIRNOR embedding + RoPE        (tests numeric embedding)
  4. SPIRNOR embedding + SPIRNOR-RoPE (tests combined SPIRNOR)

Tasks (5 number-theoretic classification tasks):
  1. GCD(a,b)        → 9 classes: {1,2,3,4,5,6,7,8,>=9}
  2. SPF(n)          → 10 classes: {2,3,5,7,11,13,17,19,23,>=29}
  3. NumDivisors(n)  → 13 classes: {1,2,...,12,>=13}
  4. Omega(n)        → 5 classes: {1,2,3,4,>=5} distinct prime factors
  5. IsCoprime(a,b)  → 2 classes: {not coprime, coprime}

Training: numbers 2-2000 (50K per task = 250K total)
Testing:  in-range + 4 OOD ranges up to 500K (1000 per task per range)

Environment: Python 3.12, PyTorch 2.10.0+cu130, RTX 3060 Ti
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import json
import time
import functools
from collections import OrderedDict
from math import gcd

# Unbuffered printing for background execution
print = functools.partial(print, flush=True)

# ============================================================
# DEVICE SETUP
# ============================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {props.total_memory / 1e9:.1f} GB")

# ============================================================
# SPIRNOR CONSTANTS
# ============================================================

PHI = (1 + math.sqrt(5)) / 2

SPIRNOR_CONSTANTS = OrderedDict([
    ('pi', math.pi),
    ('sqrt2', math.sqrt(2)),
    ('phi_sq', PHI ** 2),
    ('e', math.e),
    ('golden_angle', 2 * math.pi / PHI ** 2),
    ('phi', PHI),
    ('ln2', math.log(2)),
    ('pi_e', math.pi / math.e),
])

SPIRNOR_CONST_LIST = list(SPIRNOR_CONSTANTS.values())

# ============================================================
# NUMBER THEORY HELPERS
# ============================================================

def sieve_spf(max_n):
    """Sieve returning smallest prime factor for each n up to max_n."""
    spf = list(range(max_n + 1))
    for i in range(2, int(max_n ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, max_n + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf


def count_divisors(n):
    """Count total number of divisors of n."""
    if n <= 0:
        return 0
    count = 0
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            count += 2 if i != n // i else 1
    return count


def count_distinct_prime_factors(n):
    """Count distinct prime factors — the omega function."""
    if n <= 1:
        return 0
    count = 0
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            count += 1
            while temp % d == 0:
                temp //= d
        d += 1
    if temp > 1:
        count += 1
    return count


# ============================================================
# TASK DEFINITIONS
# ============================================================

TASKS = OrderedDict([
    ('gcd',     {'n_inputs': 2, 'n_classes': 9,  'name': 'GCD'}),
    ('spf',     {'n_inputs': 1, 'n_classes': 10, 'name': 'SPF'}),
    ('ndiv',    {'n_inputs': 1, 'n_classes': 13, 'name': 'NumDiv'}),
    ('omega',   {'n_inputs': 1, 'n_classes': 5,  'name': 'Omega'}),
    ('coprime', {'n_inputs': 2, 'n_classes': 2,  'name': 'Coprime'}),
])

TASK_LIST = list(TASKS.keys())
TASK_ID = {name: i for i, name in enumerate(TASK_LIST)}
N_TASKS = len(TASK_LIST)

SPF_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]


def gcd_to_class(g):
    return min(g - 1, 8) if g >= 1 else 0


def spf_to_class(p):
    try:
        return SPF_PRIMES.index(p)
    except ValueError:
        return len(SPF_PRIMES)  # class 9 = ">=29"


def ndiv_to_class(d):
    return min(d - 1, 12) if d >= 1 else 0


def omega_to_class(w):
    return min(w - 1, 4) if w >= 1 else 0


# ============================================================
# NUMERIC EMBEDDINGS
# ============================================================

class SPIRNORNumericEmbedding(nn.Module):
    """SPIRNOR log-spiral numeric embedding.

    Maps any positive integer to a d_model vector using the SPIRNOR equation:
      For each constant C: compute r=ln(n), theta=(C*n) mod 2pi, phi=(PHI*n) mod 2pi
      Features: [x, y, z, sin(theta), cos(theta), sin(phi), cos(phi)] per constant
    Then projects the 56-dim raw features to d_model via learned linear layer.

    Key advantage: deterministic and works for ANY integer, enabling OOD generalization.
    """
    def __init__(self, d_model, constants=None):
        super().__init__()
        consts = list((constants or SPIRNOR_CONSTANTS).values())
        self.register_buffer('const_vals',
                             torch.tensor(consts, dtype=torch.float32))
        self.n_consts = len(consts)
        raw_dim = 7 * self.n_consts  # 7 features per constant
        self.proj = nn.Linear(raw_dim, d_model)
        self.pad_embed = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(self, numbers):
        """numbers: (B, S) int tensor. 0 = padding."""
        mask = (numbers > 0)
        n_vals = numbers.float().clamp(min=1.0)

        all_feats = []
        phi_val = PHI
        for i in range(self.n_consts):
            C = self.const_vals[i]
            r = torch.log(n_vals)
            theta = (C * n_vals) % (2 * math.pi)
            phi_angle = (phi_val * n_vals) % (2 * math.pi)
            x = r * torch.sin(theta) * torch.cos(phi_angle)
            y = r * torch.sin(theta) * torch.sin(phi_angle)
            z = r * torch.cos(theta)
            feats = torch.stack([x, y, z,
                torch.sin(theta), torch.cos(theta),
                torch.sin(phi_angle), torch.cos(phi_angle)], dim=-1)
            all_feats.append(feats)

        raw = torch.cat(all_feats, dim=-1)  # (B, S, 56)
        embedded = self.proj(raw)  # (B, S, D)

        # Replace padding positions
        pad = self.pad_embed.view(1, 1, -1).expand_as(embedded)
        embedded = torch.where(mask.unsqueeze(-1), embedded, pad)
        return embedded


class LearnedNumericEmbedding(nn.Module):
    """Standard learned embedding table.
    For OOD numbers > max_num, uses modular hashing (which destroys structure).
    """
    def __init__(self, max_num, d_model):
        super().__init__()
        self.max_num = max_num
        # Index 0 serves as PAD embedding
        self.embed = nn.Embedding(max_num + 1, d_model)

    def forward(self, numbers):
        idx = numbers.long() % (self.max_num + 1)
        return self.embed(idx)


class SinusoidalNumericEmbedding(nn.Module):
    """Sinusoidal embedding based on number value (not position).
    Deterministic like SPIRNOR, but uses standard geometric frequency progression.
    """
    def __init__(self, d_model, base=10000):
        super().__init__()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(base) / d_model)
        )
        self.register_buffer('div_term', div_term)
        self.d_model = d_model
        self.proj = nn.Linear(d_model, d_model)
        self.pad_embed = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(self, numbers):
        mask = (numbers > 0)
        n = numbers.float().unsqueeze(-1)  # (B, S, 1)
        pe = torch.zeros(*numbers.shape, self.d_model, device=numbers.device)
        pe[..., 0::2] = torch.sin(n * self.div_term)
        pe[..., 1::2] = torch.cos(n * self.div_term)
        embedded = self.proj(pe)

        pad = self.pad_embed.view(1, 1, -1).expand_as(embedded)
        embedded = torch.where(mask.unsqueeze(-1), embedded, pad)
        return embedded


# ============================================================
# ROTARY POSITION ENCODINGS
# ============================================================

def apply_rotary(x, cos, sin):
    """Apply rotary embedding to x: (B, H, S, D)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    d = min(x1.size(-1), cos.size(-1))
    out1 = x1[..., :d] * cos[..., :d] - x2[..., :d] * sin[..., :d]
    out2 = x1[..., :d] * sin[..., :d] + x2[..., :d] * cos[..., :d]
    result = torch.stack([out1, out2], dim=-1).flatten(-2)
    if result.size(-1) < x.size(-1):
        result = torch.cat([result, x[..., result.size(-1):]], dim=-1)
    return result


class StandardRoPE(nn.Module):
    """Standard RoPE: geometric frequency progression from base 10000."""
    def __init__(self, d_head, max_len=32, base=10000):
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


class SPIRNORRoPE(nn.Module):
    """SPIRNOR-RoPE: winding constants as rotation frequencies."""
    def __init__(self, d_head, max_len=32, constants=None):
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


# ============================================================
# TRANSFORMER MODEL
# ============================================================

class MultiHeadAttention(nn.Module):
    """Bidirectional multi-head attention with optional rotary PE.
    Uses F.scaled_dot_product_attention for GPU optimization.
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

    def forward(self, x, rotary=None):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.nhead, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.nhead, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.nhead, self.d_head).transpose(1, 2)

        if rotary is not None:
            q = rotary(q, S)
            k = rotary(k, S)

        drop_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(out)


class TransformerEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer."""
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, rotary=None):
        x = x + self.attn(self.norm1(x), rotary)
        x = x + self.ff(self.norm2(x))
        return x


class NumericTransformer(nn.Module):
    """Transformer encoder for number-theoretic classification tasks.

    Input: task_ids (B,), numbers (B, 2)
    Sequence: [task_embed, num1_embed, num2_embed] (3 tokens)
    Output: classification logits via task-specific heads
    """
    def __init__(self, d_model, nhead, num_layers, embed_type,
                 task_configs, max_num=2000, max_len=4,
                 pe_type='rope', dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Numeric embedding
        if embed_type == 'learned':
            self.num_embed = LearnedNumericEmbedding(max_num, d_model)
        elif embed_type == 'spirnor':
            self.num_embed = SPIRNORNumericEmbedding(d_model)
        elif embed_type == 'sinusoidal':
            self.num_embed = SinusoidalNumericEmbedding(d_model)

        # Task embedding (always learned — small fixed vocabulary)
        self.task_embed = nn.Embedding(len(task_configs), d_model)

        # Position encoding
        d_head = d_model // nhead
        if pe_type == 'rope':
            self.rotary = StandardRoPE(d_head, max_len)
        elif pe_type == 'spirnor_rope':
            self.rotary = SPIRNORRoPE(d_head, max_len)
        else:
            self.rotary = None

        # Transformer encoder
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Task-specific 2-layer MLP classification heads
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, cfg['n_classes']),
            )
            for name, cfg in task_configs.items()
        })

    def forward(self, task_ids, numbers):
        """
        task_ids: (B,) int — index into TASK_LIST
        numbers: (B, 2) int — [num1, num2], num2=0 for single-input tasks
        Returns: dict of {task_name: (B, n_classes) logits}
        """
        task_emb = self.task_embed(task_ids).unsqueeze(1)  # (B, 1, D)
        num_emb = self.num_embed(numbers)  # (B, 2, D)
        x = torch.cat([task_emb, num_emb], dim=1)  # (B, 3, D)

        for layer in self.layers:
            x = layer(x, self.rotary)
        x = self.final_norm(x)

        # Use task token (position 0) for classification
        cls_repr = x[:, 0]  # (B, D)
        return {name: head(cls_repr) for name, head in self.heads.items()}


# ============================================================
# DATA GENERATION
# ============================================================

def generate_dataset(n_per_task, num_range, rng, spf_sieve):
    """Generate balanced multi-task dataset.

    Returns: task_ids (N,), numbers (N, 2), labels (N,)
    """
    lo, hi = num_range
    all_task_ids = []
    all_numbers = []
    all_labels = []

    for task_name in TASK_LIST:
        task_id = TASK_ID[task_name]
        for _ in range(n_per_task):
            a = int(rng.randint(lo, hi + 1))
            b = int(rng.randint(lo, hi + 1))

            if task_name == 'gcd':
                label = gcd_to_class(gcd(a, b))
                nums = [a, b]
            elif task_name == 'spf':
                p = spf_sieve[a] if a < len(spf_sieve) else _trial_spf(a)
                label = spf_to_class(p)
                nums = [a, 0]
            elif task_name == 'ndiv':
                label = ndiv_to_class(count_divisors(a))
                nums = [a, 0]
            elif task_name == 'omega':
                label = omega_to_class(count_distinct_prime_factors(a))
                nums = [a, 0]
            elif task_name == 'coprime':
                label = 1 if gcd(a, b) == 1 else 0
                nums = [a, b]

            all_task_ids.append(task_id)
            all_numbers.append(nums)
            all_labels.append(label)

    # Shuffle
    perm = rng.permutation(len(all_task_ids))
    task_ids = torch.tensor([all_task_ids[i] for i in perm], dtype=torch.long)
    numbers = torch.tensor([all_numbers[i] for i in perm], dtype=torch.long)
    labels = torch.tensor([all_labels[i] for i in perm], dtype=torch.long)

    return task_ids, numbers, labels


def _trial_spf(n):
    """Fallback trial division for smallest prime factor."""
    if n <= 1:
        return n
    if n % 2 == 0:
        return 2
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return i
    return n


# ============================================================
# TRAINING
# ============================================================

def train_model(model, task_ids, numbers, labels,
                epochs=30, batch_size=1024, lr=3e-4):
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    n = len(task_ids)
    n_params = sum(p.numel() for p in model.parameters())

    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            b_task_ids = task_ids[idx].to(device)
            b_numbers = numbers[idx].to(device)
            b_labels = labels[idx].to(device)

            optimizer.zero_grad()
            all_logits = model(b_task_ids, b_numbers)

            # Compute loss grouped by task
            loss = torch.tensor(0.0, device=device)
            batch_n = 0
            for task_name in TASK_LIST:
                tid = TASK_ID[task_name]
                task_mask = (b_task_ids == tid)
                if task_mask.sum() == 0:
                    continue
                task_logits = all_logits[task_name][task_mask]
                task_labels = b_labels[task_mask]
                task_loss = F.cross_entropy(task_logits, task_labels,
                                            reduction='sum')
                loss = loss + task_loss
                batch_n += task_mask.sum().item()

                preds = task_logits.argmax(dim=-1)
                total_correct += (preds == task_labels).sum().item()

            loss = loss / max(batch_n, 1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * batch_n
            total_samples += batch_n

        scheduler.step()
        avg_loss = total_loss / max(total_samples, 1)
        avg_acc = total_correct / max(total_samples, 1)

        if (epoch + 1) in [1, 5, 10, 15, 20, 25, 30]:
            print(f"    Epoch {epoch + 1:3d}/{epochs}: "
                  f"loss={avg_loss:.4f}, acc={avg_acc:.3f}")

    return n_params


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, task_ids, numbers, labels, batch_size=2048):
    """Evaluate model on a dataset. Returns per-task and overall accuracy."""
    model.eval()
    model.to(device)

    task_correct = {t: 0 for t in TASK_LIST}
    task_total = {t: 0 for t in TASK_LIST}

    for i in range(0, len(task_ids), batch_size):
        b_task_ids = task_ids[i:i + batch_size].to(device)
        b_numbers = numbers[i:i + batch_size].to(device)
        b_labels = labels[i:i + batch_size].to(device)

        all_logits = model(b_task_ids, b_numbers)

        for task_name in TASK_LIST:
            tid = TASK_ID[task_name]
            task_mask = (b_task_ids == tid)
            if task_mask.sum() == 0:
                continue
            preds = all_logits[task_name][task_mask].argmax(dim=-1)
            correct = (preds == b_labels[task_mask]).sum().item()
            task_correct[task_name] += correct
            task_total[task_name] += task_mask.sum().item()

    results = {}
    total_c, total_t = 0, 0
    for task in TASK_LIST:
        acc = task_correct[task] / max(task_total[task], 1)
        results[task] = round(acc, 4)
        total_c += task_correct[task]
        total_t += task_total[task]

    results['overall'] = round(total_c / max(total_t, 1), 4)
    return results


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment():
    print("=" * 70)
    print("SPIRNOR AI PHASE 6: TOKEN-LEVEL TRANSFORMER + SPIRNOR EMBEDDING")
    print("=" * 70)
    print()

    # Hyperparameters
    d_model = 128
    nhead = 4
    num_layers = 4
    epochs = 30
    batch_size = 1024
    lr = 3e-4
    max_len = 4

    train_range = (2, 2000)
    n_train_per_task = 50000   # 250K total
    n_eval_per_task = 1000     # 5000 per range

    test_ranges = OrderedDict([
        ('in_range',       (2, 2000)),
        ('ood_2k_5k',      (2001, 5000)),
        ('ood_5k_20k',     (5001, 20000)),
        ('ood_20k_100k',   (20001, 100000)),
        ('ood_100k_500k',  (100001, 500000)),
    ])

    # Configurations
    configs = OrderedDict([
        ('Learned+RoPE',     {'embed': 'learned',    'pe': 'rope'}),
        ('Sinusoidal+RoPE',  {'embed': 'sinusoidal', 'pe': 'rope'}),
        ('SPIRNOR+RoPE',     {'embed': 'spirnor',    'pe': 'rope'}),
        ('SPIRNOR+SRoPE',    {'embed': 'spirnor',    'pe': 'spirnor_rope'}),
    ])

    task_configs = {name: cfg for name, cfg in TASKS.items()}

    # Precompute SPF sieve
    max_eval_num = max(hi for _, hi in test_ranges.values())
    print(f"Precomputing SPF sieve up to {max_eval_num:,}...")
    t0 = time.time()
    spf_sieve = sieve_spf(max_eval_num)
    print(f"  Done in {time.time() - t0:.1f}s ({len(spf_sieve):,} entries)")

    # Generate training data
    print(f"\nGenerating training data (range {train_range})...")
    t0 = time.time()
    rng = np.random.RandomState(42)
    train_task_ids, train_numbers, train_labels = \
        generate_dataset(n_train_per_task, train_range, rng, spf_sieve)
    gen_time = time.time() - t0
    print(f"  {len(train_task_ids):,} examples ({n_train_per_task:,}/task "
          f"x {N_TASKS} tasks) in {gen_time:.1f}s")

    # Class distribution summary
    for task_name in TASK_LIST:
        mask = (train_task_ids == TASK_ID[task_name])
        task_labels = train_labels[mask]
        n_cls = TASKS[task_name]['n_classes']
        dist = [(task_labels == c).sum().item() for c in range(n_cls)]
        majority_pct = max(dist) / sum(dist) * 100
        print(f"  {TASKS[task_name]['name']:8s}: {n_cls} classes, "
              f"majority={majority_pct:.0f}%, dist={dist}")

    # Generate test data
    print(f"\nGenerating test data...")
    test_datasets = OrderedDict()
    for range_name, r in test_ranges.items():
        test_rng = np.random.RandomState(hash(range_name) % 2**31)
        t0 = time.time()
        data = generate_dataset(n_eval_per_task, r, test_rng, spf_sieve)
        gen_t = time.time() - t0
        test_datasets[range_name] = data
        print(f"  {range_name:18s}: {len(data[0]):,} examples "
              f"(range {r[0]:,}-{r[1]:,}) [{gen_t:.1f}s]")

    # ============================================================
    # TRAINING AND EVALUATION LOOP
    # ============================================================

    all_results = OrderedDict()

    print(f"\n{'=' * 70}")
    print(f"TRAINING AND EVALUATION")
    print(f"{'=' * 70}")
    print(f"Architecture: {num_layers}-layer Transformer encoder, "
          f"d_model={d_model}, {nhead} heads")
    print(f"Training: {epochs} epochs, batch_size={batch_size}, lr={lr}")

    for config_name, cfg in configs.items():
        print(f"\n{'=' * 60}")
        print(f"CONFIG: {config_name}")
        print(f"  Embedding: {cfg['embed']}, Position: {cfg['pe']}")
        print(f"{'=' * 60}")

        model = NumericTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            embed_type=cfg['embed'],
            task_configs=task_configs,
            max_num=train_range[1],
            max_len=max_len,
            pe_type=cfg['pe'],
            dropout=0.1,
        )

        t0 = time.time()
        n_params = train_model(
            model, train_task_ids, train_numbers, train_labels,
            epochs=epochs, batch_size=batch_size, lr=lr
        )
        train_time = time.time() - t0
        print(f"  Training time: {train_time:.1f}s, Params: {n_params:,}")

        # Evaluate on each range
        print(f"\n  Evaluation:")
        config_results = OrderedDict()
        config_results['params'] = n_params
        config_results['train_time'] = train_time

        for range_name, data in test_datasets.items():
            t_ids, nums, lbls = data
            results = evaluate(model, t_ids, nums, lbls)
            config_results[range_name] = results

            task_str = ' '.join(
                f"{TASKS[t]['name']}={results[t]:.3f}" for t in TASK_LIST
            )
            print(f"    {range_name:18s}: overall={results['overall']:.4f}  "
                  f"[{task_str}]")

        all_results[config_name] = config_results

        # Free GPU memory
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return all_results


# ============================================================
# RESULTS SUMMARY
# ============================================================

def print_summary(results):
    range_keys = ['in_range', 'ood_2k_5k', 'ood_5k_20k',
                  'ood_20k_100k', 'ood_100k_500k']
    range_labels = ['In-Range', '2K-5K', '5K-20K', '20K-100K', '100K-500K']

    print("\n" + "=" * 70)
    print("PHASE 6 RESULTS SUMMARY")
    print("=" * 70)

    # Overall accuracy table
    print("\n  Overall Accuracy:")
    header = f"  {'Config':20s} | {'Params':>10s}"
    for label in range_labels:
        header += f" | {label:>9s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for config_name, res in results.items():
        line = f"  {config_name:20s} | {res['params']:10,}"
        for key in range_keys:
            val = res.get(key, {}).get('overall', 0)
            best = max(r.get(key, {}).get('overall', 0) for r in results.values())
            marker = '*' if abs(val - best) < 0.001 and val > 0 else ' '
            line += f" | {val:.4f}{marker}"
        print(line)

    # Per-task breakdown
    print("\n" + "=" * 70)
    print("PER-TASK BREAKDOWN (best marked with *)")
    print("=" * 70)

    short_names = {
        'Learned+RoPE': 'Learned',
        'Sinusoidal+RoPE': 'Sine',
        'SPIRNOR+RoPE': 'SPIRNOR',
        'SPIRNOR+SRoPE': 'SP+SR',
    }

    for task_name in TASK_LIST:
        task_label = TASKS[task_name]['name']
        print(f"\n  {task_label} ({TASKS[task_name]['n_classes']} classes):")

        for key, label in zip(range_keys, range_labels):
            parts = []
            vals = []
            for config_name, res in results.items():
                v = res.get(key, {}).get(task_name, 0)
                vals.append(v)
                parts.append((short_names.get(config_name, config_name), v))

            best_val = max(vals) if vals else 0
            line_parts = []
            for cname, v in parts:
                marker = '*' if abs(v - best_val) < 0.005 and v > 0 else ' '
                line_parts.append(f"{cname}={v:.3f}{marker}")
            print(f"    {label:10s}: {'  '.join(line_parts)}")

    # SPIRNOR vs Learned head-to-head
    spirnor_res = results.get('SPIRNOR+RoPE', {})
    learned_res = results.get('Learned+RoPE', {})

    if spirnor_res and learned_res:
        print("\n" + "=" * 70)
        print("SPIRNOR vs LEARNED HEAD-TO-HEAD")
        print("=" * 70)

        s_wins, l_wins, ties = 0, 0, 0

        for task_name in TASK_LIST:
            task_label = TASKS[task_name]['name']
            print(f"\n  {task_label}:")
            for key, label in zip(range_keys, range_labels):
                s = spirnor_res.get(key, {}).get(task_name, 0)
                l = learned_res.get(key, {}).get(task_name, 0)
                diff = s - l
                if abs(diff) < 0.01:
                    winner = "TIE"
                    ties += 1
                elif diff > 0:
                    winner = "SPIRNOR"
                    s_wins += 1
                else:
                    winner = "LEARNED"
                    l_wins += 1
                print(f"    {label:10s}: SPIRNOR={s:.4f}  Learned={l:.4f}  "
                      f"diff={diff:+.4f}  ({winner})")

        total = s_wins + l_wins + ties
        print(f"\n  SCOREBOARD: SPIRNOR {s_wins}/{total} wins, "
              f"Learned {l_wins}/{total} wins, {ties} ties")

    # OOD generalization degradation
    print("\n" + "=" * 70)
    print("OOD GENERALIZATION (In-Range -> 100K-500K)")
    print("=" * 70)

    for config_name, res in results.items():
        in_acc = res.get('in_range', {}).get('overall', 0)
        far_acc = res.get('ood_100k_500k', {}).get('overall', 0)
        drop = in_acc - far_acc
        retain = (far_acc / in_acc * 100) if in_acc > 0 else 0
        print(f"  {config_name:20s}: {in_acc:.4f} -> {far_acc:.4f}  "
              f"(drop={drop:.4f}, retain={retain:.0f}%)")

    # Overall scoreboard
    print("\n" + "=" * 70)
    print("OVERALL SCOREBOARD (wins per range)")
    print("=" * 70)

    wins = {name: 0 for name in results}
    for key in range_keys:
        vals = {name: res.get(key, {}).get('overall', 0)
                for name, res in results.items()}
        best = max(vals.values())
        for name, v in vals.items():
            if abs(v - best) < 0.001 and v > 0:
                wins[name] += 1

    for name in sorted(wins, key=lambda x: -wins[x]):
        bar = '#' * (wins[name] * 5)
        print(f"  {name:20s}: {wins[name]}/{len(range_keys)} wins  {bar}")


def save_results(results):
    def clean(obj):
        if isinstance(obj, (OrderedDict, dict)):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    save_path = 'Scripts/SPIRNOR_AI_PHASE6_RESULTS.json'
    all_data = {
        'results': clean(results),
        'config': {
            'd_model': 128, 'nhead': 4, 'num_layers': 4, 'epochs': 30,
            'train_range': [2, 2000],
            'n_train_per_task': 50000,
            'n_eval_per_task': 1000,
            'tasks': {name: dict(cfg) for name, cfg in TASKS.items()},
            'configs': list(results.keys()),
            'spirnor_constants': dict(SPIRNOR_CONSTANTS),
        }
    }
    with open(save_path, 'w') as f:
        json.dump(all_data, f, indent=2)
    print(f"\nResults saved to: {save_path}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    results = run_experiment()
    print_summary(results)
    save_results(results)
