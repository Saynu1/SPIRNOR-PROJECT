#!/usr/bin/env python3
"""
SPIRNOR AI Phase 8: Rational Fraction Constants

Phase 7's visualization analysis revealed WHY pi dominates: pi = 2pi/2 creates
an exact parity encoding (mod 2). This insight generalizes: C = 2pi/p creates
an exact mod-p encoding with exactly p discrete angular positions.

Hypothesis: Constants {2pi/2, 2pi/3, 2pi/5, 2pi/7, 2pi/11} should create
EXACT modular residue encodings for the first 5 primes, outperforming the
original irrational constants which create only approximate structure.

Together, 5 prime-rational constants encode mod-2*3*5*7*11 = mod-2310 structure,
which is preserved exactly regardless of number magnitude — enabling superior
OOD generalization.

Configurations tested (7):
  1. irrational_8   : Original 8 constants (Phase 6 baseline)
  2. irrational_3   : Sweet spot (pi, sqrt2, phi^2) from Phase 7
  3. rational_5     : 2pi/2, 2pi/3, 2pi/5, 2pi/7, 2pi/11
  4. rational_8     : 2pi/2, 2pi/3, 2pi/5, 2pi/7, 2pi/11, 2pi/13, 2pi/17, 2pi/19
  5. rational_3     : 2pi/2, 2pi/3, 2pi/5  (minimal rational)
  6. hybrid_5       : 2pi/2, 2pi/3, 2pi/5 + sqrt2 + phi^2  (rational + irrational)
  7. learned_baseline: Learned embedding (Phase 6 baseline for reference)

Architecture: Identical to Phase 6 (4-layer Transformer encoder, d_model=128,
4 heads, 30 epochs, 250K training examples, 5 tasks).

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

# Original irrational constants (from Phase 6)
IRRATIONAL_CONSTANTS = OrderedDict([
    ('pi', math.pi),
    ('sqrt2', math.sqrt(2)),
    ('phi_sq', PHI ** 2),
    ('e', math.e),
    ('golden_angle', TWO_PI / PHI ** 2),
    ('phi', PHI),
    ('ln2', math.log(2)),
    ('pi_e', math.pi / math.e),
])

# Rational constants: 2pi/p for first 8 primes
# These create EXACT mod-p encodings with p discrete angular positions
RATIONAL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19]
RATIONAL_CONSTANTS_8 = OrderedDict([
    (f'2pi/{p}', TWO_PI / p) for p in RATIONAL_PRIMES
])

# Configuration definitions
CONST_CONFIGS = OrderedDict([
    ('irrational_8', {
        'names': list(IRRATIONAL_CONSTANTS.keys()),
        'values': list(IRRATIONAL_CONSTANTS.values()),
        'type': 'spirnor',
        'desc': 'Original 8 irrationals (Phase 6 baseline)',
    }),
    ('irrational_3', {
        'names': ['pi', 'sqrt2', 'phi_sq'],
        'values': [math.pi, math.sqrt(2), PHI ** 2],
        'type': 'spirnor',
        'desc': 'Sweet spot from Phase 7 (pi, sqrt2, phi^2)',
    }),
    ('rational_5', {
        'names': [f'2pi/{p}' for p in [2, 3, 5, 7, 11]],
        'values': [TWO_PI / p for p in [2, 3, 5, 7, 11]],
        'type': 'spirnor',
        'desc': 'Rational 2pi/p for first 5 primes (mod 2*3*5*7*11=2310)',
    }),
    ('rational_8', {
        'names': [f'2pi/{p}' for p in RATIONAL_PRIMES],
        'values': [TWO_PI / p for p in RATIONAL_PRIMES],
        'type': 'spirnor',
        'desc': 'Rational 2pi/p for first 8 primes (mod 9,699,690)',
    }),
    ('rational_3', {
        'names': [f'2pi/{p}' for p in [2, 3, 5]],
        'values': [TWO_PI / p for p in [2, 3, 5]],
        'type': 'spirnor',
        'desc': 'Rational 2pi/p for first 3 primes (mod 30)',
    }),
    ('hybrid_5', {
        'names': ['2pi/2', '2pi/3', '2pi/5', 'sqrt2', 'phi_sq'],
        'values': [TWO_PI / 2, TWO_PI / 3, TWO_PI / 5, math.sqrt(2), PHI ** 2],
        'type': 'spirnor',
        'desc': 'Rational primes {2,3,5} + best irrationals {sqrt2, phi^2}',
    }),
    ('learned_baseline', {
        'names': [],
        'values': [],
        'type': 'learned',
        'desc': 'Standard learned embedding (reference baseline)',
    }),
])

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
        return len(SPF_PRIMES)

def ndiv_to_class(d):
    return min(d - 1, 12) if d >= 1 else 0

def omega_to_class(w):
    return min(w - 1, 4) if w >= 1 else 0

# ============================================================
# EMBEDDINGS
# ============================================================

class SPIRNORNumericEmbedding(nn.Module):
    """Configurable SPIRNOR embedding with arbitrary constant sets."""
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

        if embed_type == 'learned':
            self.num_embed = LearnedNumericEmbedding(max_num, d_model)
        elif embed_type == 'spirnor':
            self.num_embed = SPIRNORNumericEmbedding(
                d_model, const_values, const_names)

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

def generate_dataset(n_per_task, num_range, rng, spf_sieve):
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

# ============================================================
# SINGLE CONFIG RUN
# ============================================================

def run_single_config(config_name, config, train_data, test_datasets,
                      d_model, nhead, num_layers, epochs, batch_size, lr,
                      max_num):
    print(f"\n  {'-' * 56}")
    print(f"  CONFIG: {config_name}")
    print(f"  {config['desc']}")
    if config['type'] == 'spirnor':
        print(f"  Constants ({len(config['names'])}): {', '.join(config['names'])}")
        print(f"  Values: {[f'{v:.6f}' for v in config['values']]}")
        print(f"  Raw features: {7 * len(config['names'])} -> d_model={d_model}")
    else:
        print(f"  Learned embedding (vocab={max_num})")
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
        const_values=config.get('values'),
        const_names=config.get('names'),
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
    result['n_constants'] = len(config.get('values', []))
    result['constants'] = config.get('names', [])
    result['const_type'] = config['type']

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
# MAIN EXPERIMENT
# ============================================================

def run_experiment():
    print("=" * 70)
    print("SPIRNOR AI PHASE 8: RATIONAL FRACTION CONSTANTS")
    print("Testing 2pi/p prime-modular encodings vs irrational constants")
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

    # Print constant analysis
    print("\n" + "-" * 70)
    print("CONSTANT SET ANALYSIS")
    print("-" * 70)
    for cfg_name, cfg in CONST_CONFIGS.items():
        if cfg['type'] == 'learned':
            print(f"\n  {cfg_name}: Learned embedding (no constants)")
            continue
        n_const = len(cfg['values'])
        print(f"\n  {cfg_name} ({n_const} constants):")
        for name, val in zip(cfg['names'], cfg['values']):
            # Analyze: is this a rational fraction of 2pi?
            ratio = val / TWO_PI
            # Check if ratio is close to 1/p for small p
            is_rational = False
            for p in range(2, 50):
                if abs(ratio - 1.0/p) < 1e-10:
                    print(f"    {name:15s} = {val:.6f}  (= 2pi/{p}, "
                          f"creates EXACT mod-{p} encoding)")
                    is_rational = True
                    break
            if not is_rational:
                print(f"    {name:15s} = {val:.6f}  (irrational, "
                      f"equidistributed)")

    # Precompute SPF sieve
    max_eval_num = max(hi for _, hi in test_ranges.values())
    print(f"\nPrecomputing SPF sieve up to {max_eval_num:,}...")
    t0 = time.time()
    spf_sieve = sieve_spf(max_eval_num)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Generate training data (fixed seed)
    print(f"\nGenerating training data (range {train_range})...")
    rng = np.random.RandomState(42)
    train_data = generate_dataset(n_train_per_task, train_range, rng, spf_sieve)
    print(f"  {len(train_data[0]):,} examples")

    # Generate test data (fixed seeds per range)
    print(f"Generating test data...")
    test_datasets = OrderedDict()
    for range_name, r in test_ranges.items():
        test_rng = np.random.RandomState(hash(range_name) % 2**31)
        data = generate_dataset(n_eval_per_task, r, test_rng, spf_sieve)
        test_datasets[range_name] = data
        print(f"  {range_name:18s}: {len(data[0]):,} examples "
              f"(range {r[0]:,}-{r[1]:,})")

    # Run all configurations
    print(f"\n{'=' * 70}")
    print(f"TRAINING AND EVALUATION ({len(CONST_CONFIGS)} configurations)")
    print(f"Architecture: {num_layers}-layer Transformer, d_model={d_model}, "
          f"{nhead} heads, {epochs} epochs")
    print(f"{'=' * 70}")

    all_results = OrderedDict()
    for i, (cfg_name, cfg) in enumerate(CONST_CONFIGS.items()):
        print(f"\n{'=' * 60}")
        print(f"[{i + 1}/{len(CONST_CONFIGS)}] {cfg_name}")
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

def print_summary(all_results):
    range_keys = ['in_range', 'ood_2k_5k', 'ood_5k_20k',
                  'ood_20k_100k', 'ood_100k_500k']
    range_labels = ['In-Range', '2K-5K', '5K-20K', '20K-100K', '100K-500K']

    print("\n" + "=" * 70)
    print("PHASE 8 RESULTS SUMMARY")
    print("=" * 70)

    # Overall accuracy table
    print("\n  Overall Accuracy:")
    header = f"  {'Config':18s} | {'K':>2s} | {'Type':>8s} | {'Params':>10s}"
    for label in range_labels:
        header += f" | {label:>9s}"
    header += f" | {'Avg OOD':>9s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for cfg_name, res in all_results.items():
        n_const = res.get('n_constants', 0)
        c_type = res.get('const_type', '?')
        line = (f"  {cfg_name:18s} | {n_const:2d} | {c_type:>8s} | "
                f"{res['params']:10,}")
        ood_vals = []
        for key, label in zip(range_keys, range_labels):
            val = res.get(key, {}).get('overall', 0)
            if key != 'in_range':
                ood_vals.append(val)
            best = max(r.get(key, {}).get('overall', 0)
                       for r in all_results.values())
            marker = '*' if abs(val - best) < 0.002 and val > 0 else ' '
            line += f" | {val:.4f}{marker}"
        avg_ood = sum(ood_vals) / len(ood_vals) if ood_vals else 0
        best_avg = max(
            sum(r.get(k, {}).get('overall', 0) for k in range_keys[1:]) / 4
            for r in all_results.values()
        )
        marker = '*' if abs(avg_ood - best_avg) < 0.002 else ' '
        line += f" | {avg_ood:.4f}{marker}"
        print(line)

    # Rational vs Irrational comparison
    print("\n" + "-" * 70)
    print("RATIONAL vs IRRATIONAL COMPARISON")
    print("-" * 70)

    comparison_pairs = [
        ('rational_3', 'irrational_3', '3 constants'),
        ('rational_5', 'irrational_3', '5-rational vs 3-irrational sweet spot'),
        ('rational_8', 'irrational_8', '8 constants'),
        ('hybrid_5', 'rational_5', 'hybrid vs pure rational (5 each)'),
        ('hybrid_5', 'irrational_3', 'hybrid vs irrational sweet spot'),
    ]

    for a_name, b_name, desc in comparison_pairs:
        if a_name not in all_results or b_name not in all_results:
            continue
        a_res = all_results[a_name]
        b_res = all_results[b_name]

        a_ood = [a_res.get(k, {}).get('overall', 0) for k in range_keys[1:]]
        b_ood = [b_res.get(k, {}).get('overall', 0) for k in range_keys[1:]]
        a_avg = sum(a_ood) / len(a_ood)
        b_avg = sum(b_ood) / len(b_ood)
        diff = a_avg - b_avg

        a_in = a_res.get('in_range', {}).get('overall', 0)
        b_in = b_res.get('in_range', {}).get('overall', 0)
        in_diff = a_in - b_in

        winner = a_name if diff > 0 else b_name
        print(f"\n  {desc}:")
        print(f"    {a_name:18s}: in_range={a_in:.4f}, avg_OOD={a_avg:.4f}")
        print(f"    {b_name:18s}: in_range={b_in:.4f}, avg_OOD={b_avg:.4f}")
        print(f"    Diff: in_range={in_diff:+.4f}, avg_OOD={diff:+.4f}  "
              f"-> {winner} wins OOD")

    # Per-task breakdown at 2K-5K
    print("\n" + "-" * 70)
    print("PER-TASK BREAKDOWN AT 2K-5K (nearest OOD range)")
    print("-" * 70)

    spirnor_configs = [k for k, v in all_results.items()
                       if v.get('const_type') == 'spirnor']

    for task_name in TASK_LIST:
        task_label = TASKS[task_name]['name']
        print(f"\n  {task_label}:")
        for cfg_name in list(all_results.keys()):
            res = all_results[cfg_name]
            val = res.get('ood_2k_5k', {}).get(task_name, 0)
            n_const = res.get('n_constants', 0)
            c_type = res.get('const_type', '?')
            bar_len = int(val * 40)
            bar = '#' * bar_len
            print(f"    {cfg_name:18s} ({c_type:>8s},{n_const:2d}C): "
                  f"{val:.4f} {bar}")

    # Scale invariance check
    print("\n" + "-" * 70)
    print("SCALE INVARIANCE (OOD degradation pattern)")
    print("-" * 70)

    for cfg_name, res in all_results.items():
        ood_vals = [res.get(k, {}).get('overall', 0) for k in range_keys[1:]]
        if not ood_vals or max(ood_vals) == 0:
            continue
        spread = max(ood_vals) - min(ood_vals)
        near = ood_vals[0]
        far = ood_vals[-1]
        print(f"  {cfg_name:18s}: "
              f"near={near:.4f} far={far:.4f} "
              f"spread={spread:.4f} "
              f"{'(FLAT - scale invariant)' if spread < 0.02 else ''}")


def save_results(all_results):
    def clean(obj):
        if isinstance(obj, (OrderedDict, dict)):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    save_data = {
        'phase': 'Phase 8: Rational Fraction Constants',
        'hypothesis': ('Constants C=2pi/p for prime p create exact mod-p '
                       'encodings that may outperform irrational constants'),
        'config': {
            'd_model': 128, 'nhead': 4, 'num_layers': 4, 'epochs': 30,
            'train_range': [2, 2000],
            'n_train_per_task': 50000,
            'n_eval_per_task': 1000,
        },
        'constant_sets': {
            name: {
                'names': cfg['names'],
                'values': cfg['values'],
                'type': cfg['type'],
                'desc': cfg['desc'],
            }
            for name, cfg in CONST_CONFIGS.items()
        },
        'results': clean(all_results),
    }

    save_path = 'Scripts/SPIRNOR_AI_PHASE8_RESULTS.json'
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {save_path}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    results = run_experiment()
    print_summary(results)
    save_results(results)
