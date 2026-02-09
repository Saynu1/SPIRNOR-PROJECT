#!/usr/bin/env python3
"""
SPIRNOR AI Phase 9: Hybrid Architectures + New Task Domains + Lightweight Scaling

Building on Phase 8's breakthrough (rational fraction constants C=2pi/p yield
75.0% avg OOD vs 58.1% irrational), this phase tests:

Part A: Hybrid Architectures (4 embedding configs, d_model=128)
  1. Pure Learned — nn.Embedding baseline
  2. Pure SPIRNOR — rational_5 {2pi/2,3,5,7,11}
  3. Additive Hybrid — spirnor(n) + learned(n) -> LayerNorm
  4. Gated Hybrid — g * spirnor(n) + (1-g) * learned(n), gate = sigmoid(MLP)

Part B: New Tasks (3 new + 5 existing = 8 total)
  - IsPrime(n): Primality testing (2 classes)
  - ModMul7(a,b): a*b mod 7 (7 classes) — exact mod-p encoding test
  - Compare(a,b): a<b, a==b, a>b (3 classes) — pure magnitude, no modular structure

Part C: Scaling Test (d_model=256, best 3 configs)
  - Verify advantage persists at larger model scale

Training: 400K examples (50K per task x 8 tasks), train 2-2000, 30 epochs
Eval ranges: in_range, 2K-5K, 5K-20K, 20K-100K, 100K-500K (1K per task per range)

Self-contained script for Vast.ai cloud deployment.
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
# CONSTANTS
# ============================================================

PHI = (1 + math.sqrt(5)) / 2
TWO_PI = 2 * math.pi

# Rational constants: 2pi/p for first 5 primes (Phase 8 winner)
RATIONAL_5_PRIMES = [2, 3, 5, 7, 11]
RATIONAL_5_VALUES = [TWO_PI / p for p in RATIONAL_5_PRIMES]
RATIONAL_5_NAMES = [f'2pi/{p}' for p in RATIONAL_5_PRIMES]

# ============================================================
# TASK DEFINITIONS (5 existing + 3 new = 8 total)
# ============================================================

TASKS = OrderedDict([
    ('gcd',      {'n_inputs': 2, 'n_classes': 9,  'name': 'GCD'}),
    ('spf',      {'n_inputs': 1, 'n_classes': 10, 'name': 'SPF'}),
    ('ndiv',     {'n_inputs': 1, 'n_classes': 13, 'name': 'NumDiv'}),
    ('omega',    {'n_inputs': 1, 'n_classes': 5,  'name': 'Omega'}),
    ('coprime',  {'n_inputs': 2, 'n_classes': 2,  'name': 'Coprime'}),
    ('isprime',  {'n_inputs': 1, 'n_classes': 2,  'name': 'IsPrime'}),
    ('modmul7',  {'n_inputs': 2, 'n_classes': 7,  'name': 'ModMul7'}),
    ('compare',  {'n_inputs': 2, 'n_classes': 3,  'name': 'Compare'}),
])

TASK_LIST = list(TASKS.keys())
TASK_ID = {name: i for i, name in enumerate(TASK_LIST)}
N_TASKS = len(TASK_LIST)
SPF_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

# ============================================================
# NUMBER THEORY HELPERS
# ============================================================

def sieve_spf(max_n):
    spf = list(range(max_n + 1))
    for i in range(2, int(max_n ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, max_n + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def count_divisors(n):
    if n <= 0:
        return 0
    count = 0
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            count += 2 if i != n // i else 1
    return count

def count_distinct_prime_factors(n):
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

def _trial_spf(n):
    if n <= 1:
        return n
    if n % 2 == 0:
        return 2
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return i
    return n

def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

# Label encoding functions
def gcd_to_class(g):
    return min(g - 1, 8) if g >= 1 else 0

def spf_to_class(p):
    try:
        return SPF_PRIMES.index(p)
    except ValueError:
        return len(SPF_PRIMES)

def ndiv_to_class(d):
    return min(d - 1, 12) if d >= 1 else 0

def omega_to_class(w):
    return min(w - 1, 4) if w >= 1 else 0

# ============================================================
# EMBEDDINGS
# ============================================================

class SPIRNORNumericEmbedding(nn.Module):
    """SPIRNOR embedding with rational fraction constants."""
    def __init__(self, d_model, const_values, const_names=None):
        super().__init__()
        self.const_names = const_names or [f'C{i}' for i in range(len(const_values))]
        self.register_buffer('const_vals',
                             torch.tensor(const_values, dtype=torch.float32))
        self.n_consts = len(const_values)
        raw_dim = 7 * self.n_consts
        self.proj = nn.Linear(raw_dim, d_model)
        self.pad_embed = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(self, numbers):
        mask = (numbers > 0)
        n_vals = numbers.float().clamp(min=1.0)

        all_feats = []
        phi_val = PHI
        for i in range(self.n_consts):
            C = self.const_vals[i]
            r = torch.log(n_vals)
            theta = (C * n_vals) % TWO_PI
            phi_angle = (phi_val * n_vals) % TWO_PI
            x = r * torch.sin(theta) * torch.cos(phi_angle)
            y = r * torch.sin(theta) * torch.sin(phi_angle)
            z = r * torch.cos(theta)
            feats = torch.stack([x, y, z,
                torch.sin(theta), torch.cos(theta),
                torch.sin(phi_angle), torch.cos(phi_angle)], dim=-1)
            all_feats.append(feats)

        raw = torch.cat(all_feats, dim=-1)
        embedded = self.proj(raw)

        pad = self.pad_embed.view(1, 1, -1).expand_as(embedded)
        embedded = torch.where(mask.unsqueeze(-1), embedded, pad)
        return embedded


class LearnedNumericEmbedding(nn.Module):
    def __init__(self, max_num, d_model):
        super().__init__()
        self.max_num = max_num
        self.embed = nn.Embedding(max_num + 1, d_model)

    def forward(self, numbers):
        idx = numbers.long() % (self.max_num + 1)
        return self.embed(idx)


class AdditiveHybridEmbedding(nn.Module):
    """Additive hybrid: spirnor(n) + learned(n) -> LayerNorm"""
    def __init__(self, max_num, d_model, const_values, const_names=None):
        super().__init__()
        self.spirnor = SPIRNORNumericEmbedding(d_model, const_values, const_names)
        self.learned = LearnedNumericEmbedding(max_num, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, numbers):
        s = self.spirnor(numbers)
        l = self.learned(numbers)
        return self.norm(s + l)


class GatedHybridEmbedding(nn.Module):
    """Gated hybrid: g * spirnor(n) + (1-g) * learned(n)
    Gate = sigmoid(MLP(concat(spirnor, learned)))
    Gate initialized balanced (bias=0 -> sigmoid(0)=0.5)
    """
    def __init__(self, max_num, d_model, const_values, const_names=None):
        super().__init__()
        self.spirnor = SPIRNORNumericEmbedding(d_model, const_values, const_names)
        self.learned = LearnedNumericEmbedding(max_num, d_model)
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        # Initialize gate balanced
        nn.init.zeros_(self.gate_mlp[-1].bias)

    def forward(self, numbers):
        s = self.spirnor(numbers)
        l = self.learned(numbers)
        concat = torch.cat([s, l], dim=-1)
        g = torch.sigmoid(self.gate_mlp(concat))  # [B, seq, 1]
        return g * s + (1 - g) * l

    def get_gate_stats(self, numbers):
        """Return mean gate value for analysis."""
        with torch.no_grad():
            s = self.spirnor(numbers)
            l = self.learned(numbers)
            concat = torch.cat([s, l], dim=-1)
            g = torch.sigmoid(self.gate_mlp(concat))
            return g.mean().item()


# ============================================================
# ROTARY POSITION ENCODING
# ============================================================

def apply_rotary(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    d = min(x1.size(-1), cos.size(-1))
    out1 = x1[..., :d] * cos[..., :d] - x2[..., :d] * sin[..., :d]
    out2 = x1[..., :d] * sin[..., :d] + x2[..., :d] * cos[..., :d]
    result = torch.stack([out1, out2], dim=-1).flatten(-2)
    if result.size(-1) < x.size(-1):
        result = torch.cat([result, x[..., result.size(-1):]], dim=-1)
    return result


class StandardRoPE(nn.Module):
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


# ============================================================
# TRANSFORMER MODEL
# ============================================================

class MultiHeadAttention(nn.Module):
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
    def __init__(self, d_model, nhead, num_layers, embed_type,
                 task_configs, max_num=2000, max_len=4,
                 dropout=0.1, const_values=None, const_names=None):
        super().__init__()
        self.d_model = d_model
        self.embed_type = embed_type

        if embed_type == 'learned':
            self.num_embed = LearnedNumericEmbedding(max_num, d_model)
        elif embed_type == 'spirnor':
            self.num_embed = SPIRNORNumericEmbedding(
                d_model, const_values, const_names)
        elif embed_type == 'additive_hybrid':
            self.num_embed = AdditiveHybridEmbedding(
                max_num, d_model, const_values, const_names)
        elif embed_type == 'gated_hybrid':
            self.num_embed = GatedHybridEmbedding(
                max_num, d_model, const_values, const_names)

        self.task_embed = nn.Embedding(len(task_configs), d_model)
        d_head = d_model // nhead
        self.rotary = StandardRoPE(d_head, max_len)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, cfg['n_classes']),
            )
            for name, cfg in task_configs.items()
        })

    def forward(self, task_ids, numbers):
        task_emb = self.task_embed(task_ids).unsqueeze(1)
        num_emb = self.num_embed(numbers)
        x = torch.cat([task_emb, num_emb], dim=1)
        for layer in self.layers:
            x = layer(x, self.rotary)
        x = self.final_norm(x)
        cls_repr = x[:, 0]
        return {name: head(cls_repr) for name, head in self.heads.items()}


# ============================================================
# DATA GENERATION
# ============================================================

def generate_primes_in_range(lo, hi, count):
    """Generate balanced prime/composite samples for IsPrime task."""
    primes = []
    composites = []

    # Collect primes and composites in range
    # For large ranges, use random sampling with primality testing
    if hi <= 500000:
        # Sample randomly and test primality
        rng_local = np.random.RandomState(hash((lo, hi, count)) % 2**31)
        candidates = set()
        attempts = 0
        max_attempts = count * 50

        while (len(primes) < count // 2 or len(composites) < count // 2) and attempts < max_attempts:
            n = int(rng_local.randint(lo, hi + 1))
            if n in candidates:
                attempts += 1
                continue
            candidates.add(n)
            if is_prime(n):
                if len(primes) < count // 2:
                    primes.append(n)
            else:
                if len(composites) < count // 2:
                    composites.append(n)
            attempts += 1

        # Fill remainder if needed
        while len(primes) < count // 2 and len(composites) > 0:
            primes.append(composites.pop())
        while len(composites) < count - len(primes):
            composites.append(int(rng_local.randint(lo, hi + 1)))

    return primes, composites


def generate_dataset(n_per_task, num_range, rng, spf_sieve):
    """Generate training/eval data for all 8 tasks."""
    lo, hi = num_range
    all_task_ids = []
    all_numbers = []
    all_labels = []

    for task_name in TASK_LIST:
        task_id = TASK_ID[task_name]

        if task_name == 'isprime':
            # Balanced sampling: 50% primes, 50% composites
            n_each = n_per_task // 2
            prime_count = 0
            composite_count = 0
            attempts = 0
            max_attempts = n_per_task * 50

            while (prime_count < n_each or composite_count < n_each) and attempts < max_attempts:
                a = int(rng.randint(lo, hi + 1))
                p = is_prime(a)
                if p and prime_count < n_each:
                    all_task_ids.append(task_id)
                    all_numbers.append([a, 0])
                    all_labels.append(1)
                    prime_count += 1
                elif not p and composite_count < n_each:
                    all_task_ids.append(task_id)
                    all_numbers.append([a, 0])
                    all_labels.append(0)
                    composite_count += 1
                attempts += 1

            # If we couldn't balance perfectly, fill remainder
            while prime_count + composite_count < n_per_task:
                a = int(rng.randint(lo, hi + 1))
                all_task_ids.append(task_id)
                all_numbers.append([a, 0])
                all_labels.append(1 if is_prime(a) else 0)
                if is_prime(a):
                    prime_count += 1
                else:
                    composite_count += 1

        elif task_name == 'compare':
            # Balanced sampling: 1/3 each for <, ==, >
            # Explicitly generate each class (a==b is rare by random sampling)
            n_each = n_per_task // 3
            n_extra = n_per_task - 3 * n_each

            # Class 0: a < b
            for _ in range(n_each + (1 if n_extra > 0 else 0)):
                a = int(rng.randint(lo, hi + 1))
                b = int(rng.randint(a + 1, hi + 2))  # b > a
                b = min(b, hi)
                if a >= b:
                    a, b = lo, lo + 1
                all_task_ids.append(task_id)
                all_numbers.append([a, b])
                all_labels.append(0)

            # Class 1: a == b
            for _ in range(n_each + (1 if n_extra > 1 else 0)):
                a = int(rng.randint(lo, hi + 1))
                all_task_ids.append(task_id)
                all_numbers.append([a, a])
                all_labels.append(1)

            # Class 2: a > b
            for _ in range(n_each):
                a = int(rng.randint(lo, hi + 1))
                b = int(rng.randint(lo, a + 1))  # b <= a
                b = max(b, lo)
                if b >= a:
                    a, b = lo + 1, lo
                all_task_ids.append(task_id)
                all_numbers.append([a, b])
                all_labels.append(2)

        elif task_name == 'modmul7':
            for _ in range(n_per_task):
                a = int(rng.randint(lo, hi + 1))
                b = int(rng.randint(lo, hi + 1))
                label = (a * b) % 7
                all_task_ids.append(task_id)
                all_numbers.append([a, b])
                all_labels.append(label)

        else:
            # Original 5 tasks
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

    perm = rng.permutation(len(all_task_ids))
    task_ids = torch.tensor([all_task_ids[i] for i in perm], dtype=torch.long)
    numbers = torch.tensor([all_numbers[i] for i in perm], dtype=torch.long)
    labels = torch.tensor([all_labels[i] for i in perm], dtype=torch.long)
    return task_ids, numbers, labels


# ============================================================
# TRAINING & EVALUATION
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


@torch.no_grad()
def evaluate(model, task_ids, numbers, labels, batch_size=2048):
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


@torch.no_grad()
def get_gate_stats(model, test_datasets):
    """Extract gate statistics from gated hybrid model."""
    if not hasattr(model, 'num_embed') or not isinstance(model.num_embed, GatedHybridEmbedding):
        return None

    model.eval()
    stats = {}
    for range_name, (t_ids, nums, lbls) in test_datasets.items():
        nums_dev = nums.to(device)
        g_val = model.num_embed.get_gate_stats(nums_dev)
        stats[range_name] = round(g_val, 4)

    return stats


# ============================================================
# EMBEDDING CONFIGS
# ============================================================

EMBED_CONFIGS = OrderedDict([
    ('learned', {
        'type': 'learned',
        'desc': 'Pure learned embedding (nn.Embedding baseline)',
        'const_values': None,
        'const_names': None,
    }),
    ('spirnor_rational5', {
        'type': 'spirnor',
        'desc': 'Pure SPIRNOR with rational_5 {2pi/2,3,5,7,11}',
        'const_values': RATIONAL_5_VALUES,
        'const_names': RATIONAL_5_NAMES,
    }),
    ('additive_hybrid', {
        'type': 'additive_hybrid',
        'desc': 'Additive: spirnor(n) + learned(n) -> LayerNorm',
        'const_values': RATIONAL_5_VALUES,
        'const_names': RATIONAL_5_NAMES,
    }),
    ('gated_hybrid', {
        'type': 'gated_hybrid',
        'desc': 'Gated: g*spirnor + (1-g)*learned, g=sigmoid(MLP)',
        'const_values': RATIONAL_5_VALUES,
        'const_names': RATIONAL_5_NAMES,
    }),
])


# ============================================================
# SINGLE CONFIG RUN
# ============================================================

def run_single_config(config_name, config, train_data, test_datasets,
                      d_model, nhead, num_layers, epochs, batch_size, lr,
                      max_num):
    print(f"\n  {'-' * 56}")
    print(f"  CONFIG: {config_name}")
    print(f"  {config['desc']}")
    print(f"  d_model={d_model}, nhead={nhead}")
    print(f"  {'-' * 56}")

    task_configs = {name: cfg for name, cfg in TASKS.items()}
    model = NumericTransformer(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        embed_type=config['type'],
        task_configs=task_configs,
        max_num=max_num,
        max_len=4,
        dropout=0.1,
        const_values=config.get('const_values'),
        const_names=config.get('const_names'),
    )

    train_task_ids, train_numbers, train_labels = train_data

    t0 = time.time()
    n_params = train_model(model, train_task_ids, train_numbers, train_labels,
                           epochs=epochs, batch_size=batch_size, lr=lr)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s, Params: {n_params:,}")

    result = OrderedDict()
    result['params'] = n_params
    result['train_time'] = round(train_time, 1)
    result['embed_type'] = config['type']
    result['d_model'] = d_model
    result['nhead'] = nhead

    # Get gate stats if gated hybrid
    gate_stats = get_gate_stats(model, test_datasets)
    if gate_stats is not None:
        result['gate_stats'] = gate_stats
        print(f"\n  Gate statistics (mean gate value, higher=more SPIRNOR):")
        for rname, gval in gate_stats.items():
            print(f"    {rname:18s}: gate={gval:.4f}")

    print(f"\n  Evaluation:")
    for range_name, data in test_datasets.items():
        t_ids, nums, lbls = data
        res = evaluate(model, t_ids, nums, lbls)
        result[range_name] = res
        task_str = ' '.join(
            f"{TASKS[t]['name']}={res[t]:.3f}" for t in TASK_LIST
        )
        print(f"    {range_name:18s}: overall={res['overall']:.4f}  "
              f"[{task_str}]")

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return result


# ============================================================
# PART A+B: MAIN EXPERIMENT (d_model=128, 4 configs, 8 tasks)
# ============================================================

def run_part_ab():
    print("=" * 70)
    print("PHASE 9 PART A+B: HYBRID ARCHITECTURES + NEW TASKS")
    print("4 embedding configs x 8 tasks, d_model=128")
    print("=" * 70)

    d_model = 128
    nhead = 4
    num_layers = 4
    epochs = 30
    batch_size = 1024
    lr = 3e-4

    train_range = (2, 2000)
    n_train_per_task = 50000
    n_eval_per_task = 1000

    test_ranges = OrderedDict([
        ('in_range',       (2, 2000)),
        ('ood_2k_5k',      (2001, 5000)),
        ('ood_5k_20k',     (5001, 20000)),
        ('ood_20k_100k',   (20001, 100000)),
        ('ood_100k_500k',  (100001, 500000)),
    ])

    # Precompute SPF sieve
    max_eval_num = max(hi for _, hi in test_ranges.values())
    print(f"\nPrecomputing SPF sieve up to {max_eval_num:,}...")
    t0 = time.time()
    spf_sieve = sieve_spf(max_eval_num)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Generate training data (fixed seed)
    print(f"\nGenerating training data (range {train_range}, "
          f"{n_train_per_task} per task x {N_TASKS} tasks)...")
    rng = np.random.RandomState(42)
    train_data = generate_dataset(n_train_per_task, train_range, rng, spf_sieve)
    print(f"  {len(train_data[0]):,} examples")

    # Verify task balance
    task_counts = {}
    for tid in train_data[0].numpy():
        task_counts[tid] = task_counts.get(tid, 0) + 1
    for task_name in TASK_LIST:
        tid = TASK_ID[task_name]
        print(f"    {task_name:10s}: {task_counts.get(tid, 0):,} examples")

    # Generate test data (fixed seeds per range)
    print(f"\nGenerating test data...")
    test_datasets = OrderedDict()
    for range_name, r in test_ranges.items():
        test_rng = np.random.RandomState(hash(range_name) % 2**31)
        data = generate_dataset(n_eval_per_task, r, test_rng, spf_sieve)
        test_datasets[range_name] = data
        print(f"  {range_name:18s}: {len(data[0]):,} examples "
              f"(range {r[0]:,}-{r[1]:,})")

    # Run all configurations
    print(f"\n{'=' * 70}")
    print(f"TRAINING AND EVALUATION ({len(EMBED_CONFIGS)} configurations)")
    print(f"Architecture: {num_layers}-layer Transformer, d_model={d_model}, "
          f"{nhead} heads, {epochs} epochs")
    print(f"{'=' * 70}")

    all_results = OrderedDict()
    for i, (cfg_name, cfg) in enumerate(EMBED_CONFIGS.items()):
        print(f"\n{'=' * 60}")
        print(f"[{i + 1}/{len(EMBED_CONFIGS)}] {cfg_name}")
        print(f"{'=' * 60}")

        result = run_single_config(
            cfg_name, cfg, train_data, test_datasets,
            d_model, nhead, num_layers, epochs, batch_size, lr,
            max_num=train_range[1]
        )
        all_results[cfg_name] = result

    return all_results, test_datasets, spf_sieve


# ============================================================
# PART C: SCALING TEST (d_model=256)
# ============================================================

def run_part_c(spf_sieve):
    print("\n" + "=" * 70)
    print("PHASE 9 PART C: SCALING TEST (d_model=256)")
    print("Testing 3 configs at larger model scale")
    print("=" * 70)

    d_model = 256
    nhead = 8
    num_layers = 4
    epochs = 30
    batch_size = 512  # smaller batch for larger model
    lr = 3e-4

    train_range = (2, 2000)
    n_train_per_task = 50000
    n_eval_per_task = 1000

    test_ranges = OrderedDict([
        ('in_range',       (2, 2000)),
        ('ood_2k_5k',      (2001, 5000)),
        ('ood_5k_20k',     (5001, 20000)),
        ('ood_20k_100k',   (20001, 100000)),
        ('ood_100k_500k',  (100001, 500000)),
    ])

    # Generate training data (same seed as Part A+B for consistency)
    print(f"\nGenerating training data for scaling test...")
    rng = np.random.RandomState(42)
    train_data = generate_dataset(n_train_per_task, train_range, rng, spf_sieve)
    print(f"  {len(train_data[0]):,} examples")

    print(f"Generating test data...")
    test_datasets = OrderedDict()
    for range_name, r in test_ranges.items():
        test_rng = np.random.RandomState(hash(range_name) % 2**31)
        data = generate_dataset(n_eval_per_task, r, test_rng, spf_sieve)
        test_datasets[range_name] = data

    # Only test 3 configs at scale: learned, spirnor, gated_hybrid
    scale_configs = OrderedDict([
        ('learned_256', EMBED_CONFIGS['learned']),
        ('spirnor_256', EMBED_CONFIGS['spirnor_rational5']),
        ('gated_hybrid_256', EMBED_CONFIGS['gated_hybrid']),
    ])

    all_results = OrderedDict()
    for i, (cfg_name, cfg) in enumerate(scale_configs.items()):
        print(f"\n{'=' * 60}")
        print(f"[{i + 1}/{len(scale_configs)}] {cfg_name} (d_model=256)")
        print(f"{'=' * 60}")

        result = run_single_config(
            cfg_name, cfg, train_data, test_datasets,
            d_model, nhead, num_layers, epochs, batch_size, lr,
            max_num=train_range[1]
        )
        all_results[cfg_name] = result

    return all_results


# ============================================================
# RESULTS SUMMARY
# ============================================================

def print_summary(ab_results, c_results):
    range_keys = ['in_range', 'ood_2k_5k', 'ood_5k_20k',
                  'ood_20k_100k', 'ood_100k_500k']
    range_labels = ['In-Range', '2K-5K', '5K-20K', '20K-100K', '100K-500K']

    print("\n" + "=" * 70)
    print("PHASE 9 RESULTS SUMMARY")
    print("=" * 70)

    # ---- Part A+B: Overall accuracy table ----
    print("\n  PART A+B: Overall Accuracy (d_model=128)")
    header = f"  {'Config':20s} | {'Params':>10s}"
    for label in range_labels:
        header += f" | {label:>9s}"
    header += f" | {'Avg OOD':>9s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for cfg_name, res in ab_results.items():
        line = f"  {cfg_name:20s} | {res['params']:10,}"
        ood_vals = []
        for key in range_keys:
            val = res.get(key, {}).get('overall', 0)
            if key != 'in_range':
                ood_vals.append(val)
            line += f" | {val:.4f}   "
        avg_ood = sum(ood_vals) / len(ood_vals) if ood_vals else 0
        line += f" | {avg_ood:.4f}   "
        print(line)

    # ---- Hybrid Analysis ----
    print("\n" + "-" * 70)
    print("HYBRID ARCHITECTURE ANALYSIS")
    print("-" * 70)

    for cfg_name in ['learned', 'spirnor_rational5', 'additive_hybrid', 'gated_hybrid']:
        if cfg_name not in ab_results:
            continue
        res = ab_results[cfg_name]
        in_acc = res.get('in_range', {}).get('overall', 0)
        ood_vals = [res.get(k, {}).get('overall', 0) for k in range_keys[1:]]
        avg_ood = sum(ood_vals) / len(ood_vals)
        print(f"  {cfg_name:20s}: in_range={in_acc:.4f}, avg_OOD={avg_ood:.4f}, "
              f"params={res['params']:,}")

    # ---- Gate Statistics ----
    if 'gated_hybrid' in ab_results and 'gate_stats' in ab_results['gated_hybrid']:
        print("\n  Gated Hybrid Gate Values (higher = more SPIRNOR):")
        for rname, gval in ab_results['gated_hybrid']['gate_stats'].items():
            print(f"    {rname:18s}: {gval:.4f}")

    # ---- New Tasks Analysis ----
    new_tasks = ['isprime', 'modmul7', 'compare']
    old_tasks = ['gcd', 'spf', 'ndiv', 'omega', 'coprime']

    print("\n" + "-" * 70)
    print("NEW TASK ANALYSIS (2K-5K OOD)")
    print("-" * 70)

    for task_name in new_tasks:
        task_label = TASKS[task_name]['name']
        print(f"\n  {task_label}:")
        for cfg_name, res in ab_results.items():
            val = res.get('ood_2k_5k', {}).get(task_name, 0)
            bar_len = int(val * 40)
            bar = '#' * bar_len
            print(f"    {cfg_name:20s}: {val:.4f} {bar}")

    # ---- Per-Task Breakdown at 2K-5K ----
    print("\n" + "-" * 70)
    print("FULL PER-TASK BREAKDOWN AT 2K-5K")
    print("-" * 70)

    task_header = f"  {'Config':20s}"
    for task_name in TASK_LIST:
        task_header += f" | {TASKS[task_name]['name']:>7s}"
    print(task_header)
    print("  " + "-" * (len(task_header) - 2))

    for cfg_name, res in ab_results.items():
        line = f"  {cfg_name:20s}"
        for task_name in TASK_LIST:
            val = res.get('ood_2k_5k', {}).get(task_name, 0)
            line += f" | {val:.4f} "
        print(line)

    # ---- Part C: Scaling Results ----
    if c_results:
        print("\n" + "=" * 70)
        print("PART C: SCALING TEST (d_model=256)")
        print("=" * 70)

        header = f"  {'Config':20s} | {'Params':>10s}"
        for label in range_labels:
            header += f" | {label:>9s}"
        header += f" | {'Avg OOD':>9s}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for cfg_name, res in c_results.items():
            line = f"  {cfg_name:20s} | {res['params']:10,}"
            ood_vals = []
            for key in range_keys:
                val = res.get(key, {}).get('overall', 0)
                if key != 'in_range':
                    ood_vals.append(val)
                line += f" | {val:.4f}   "
            avg_ood = sum(ood_vals) / len(ood_vals) if ood_vals else 0
            line += f" | {avg_ood:.4f}   "
            print(line)

        # Scaling comparison
        print("\n  Scaling Comparison (d_model=128 -> 256):")
        scale_pairs = [
            ('learned', 'learned_256'),
            ('spirnor_rational5', 'spirnor_256'),
            ('gated_hybrid', 'gated_hybrid_256'),
        ]
        for base, scaled in scale_pairs:
            if base not in ab_results or scaled not in c_results:
                continue
            b_ood = [ab_results[base].get(k, {}).get('overall', 0) for k in range_keys[1:]]
            s_ood = [c_results[scaled].get(k, {}).get('overall', 0) for k in range_keys[1:]]
            b_avg = sum(b_ood) / len(b_ood)
            s_avg = sum(s_ood) / len(s_ood)
            diff = s_avg - b_avg
            print(f"    {base:20s}: {b_avg:.4f} -> {s_avg:.4f} ({diff:+.4f})")

    # ---- Key Findings ----
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Best config overall
    best_ood_cfg = None
    best_ood_val = 0
    for cfg_name, res in ab_results.items():
        ood_vals = [res.get(k, {}).get('overall', 0) for k in range_keys[1:]]
        avg = sum(ood_vals) / len(ood_vals) if ood_vals else 0
        if avg > best_ood_val:
            best_ood_val = avg
            best_ood_cfg = cfg_name

    best_in_cfg = None
    best_in_val = 0
    for cfg_name, res in ab_results.items():
        val = res.get('in_range', {}).get('overall', 0)
        if val > best_in_val:
            best_in_val = val
            best_in_cfg = cfg_name

    print(f"\n  1. Best in-range:  {best_in_cfg} ({best_in_val:.4f})")
    print(f"  2. Best avg OOD:   {best_ood_cfg} ({best_ood_val:.4f})")

    # ModMul7 analysis (exact mod-7 test)
    if 'spirnor_rational5' in ab_results and 'learned' in ab_results:
        s_mm7 = ab_results['spirnor_rational5'].get('ood_2k_5k', {}).get('modmul7', 0)
        l_mm7 = ab_results['learned'].get('ood_2k_5k', {}).get('modmul7', 0)
        print(f"  3. ModMul7 @ 2K-5K: SPIRNOR={s_mm7:.4f}, Learned={l_mm7:.4f} "
              f"(ratio={s_mm7/max(l_mm7, 0.001):.1f}x)")

    # Compare analysis (magnitude task, no modular structure)
    if 'spirnor_rational5' in ab_results and 'learned' in ab_results:
        s_cmp = ab_results['spirnor_rational5'].get('ood_2k_5k', {}).get('compare', 0)
        l_cmp = ab_results['learned'].get('ood_2k_5k', {}).get('compare', 0)
        print(f"  4. Compare @ 2K-5K: SPIRNOR={s_cmp:.4f}, Learned={l_cmp:.4f} "
              f"(SPIRNOR should NOT dominate here — pure magnitude)")

    # Hybrid "best of both worlds" check
    if 'gated_hybrid' in ab_results:
        gh = ab_results['gated_hybrid']
        gh_in = gh.get('in_range', {}).get('overall', 0)
        gh_ood = [gh.get(k, {}).get('overall', 0) for k in range_keys[1:]]
        gh_avg_ood = sum(gh_ood) / len(gh_ood)
        spirnor_ood = best_ood_val if best_ood_cfg == 'spirnor_rational5' else 0
        learned_in = ab_results.get('learned', {}).get('in_range', {}).get('overall', 0)
        print(f"  5. Gated Hybrid: in_range={gh_in:.4f} (vs learned {learned_in:.4f}), "
              f"avg_OOD={gh_avg_ood:.4f} (vs spirnor {spirnor_ood:.4f})")
        if gh_in >= 0.97 and gh_avg_ood >= 0.70:
            print(f"     -> SUCCESS: Best of both worlds!")
        else:
            print(f"     -> Target: >=0.97 in-range AND >=0.70 OOD")


def save_results(ab_results, c_results):
    def clean(obj):
        if isinstance(obj, (OrderedDict, dict)):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    save_data = {
        'phase': 'Phase 9: Hybrid Architectures + New Task Domains + Scaling',
        'config': {
            'part_ab': {
                'd_model': 128, 'nhead': 4, 'num_layers': 4, 'epochs': 30,
                'train_range': [2, 2000], 'n_train_per_task': 50000,
                'n_eval_per_task': 1000, 'n_tasks': N_TASKS,
                'tasks': list(TASK_LIST),
            },
            'part_c': {
                'd_model': 256, 'nhead': 8, 'num_layers': 4, 'epochs': 30,
                'train_range': [2, 2000], 'n_train_per_task': 50000,
                'n_eval_per_task': 1000,
            },
        },
        'embedding_configs': {
            name: {
                'type': cfg['type'],
                'desc': cfg['desc'],
                'const_names': cfg.get('const_names'),
            }
            for name, cfg in EMBED_CONFIGS.items()
        },
        'results_part_ab': clean(ab_results),
        'results_part_c': clean(c_results) if c_results else {},
    }

    save_path = 'SPIRNOR_AI_PHASE9_RESULTS.json'
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {save_path}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    total_start = time.time()

    # Part A+B: 4 configs x 8 tasks at d_model=128
    ab_results, test_datasets, spf_sieve = run_part_ab()

    # Part C: 3 configs at d_model=256
    c_results = run_part_c(spf_sieve)

    # Summary and save
    print_summary(ab_results, c_results)
    save_results(ab_results, c_results)

    total_time = time.time() - total_start
    print(f"\nTotal experiment time: {total_time / 60:.1f} minutes")
    print("Phase 9 complete!")
