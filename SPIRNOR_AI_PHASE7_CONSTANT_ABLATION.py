#!/usr/bin/env python3
"""
SPIRNOR AI Phase 7: Constant Ablation at Transformer Scale

Building on Phase 6's decisive SPIRNOR result, this experiment asks:
which of the 8 SPIRNOR constants are actually important at scale?

Two ablation studies:
  A. Leave-One-Out: Remove each constant individually, measure impact
  B. Cumulative Addition: Add constants one by one, measure gains

This answers:
  1. Does pi remain dominant at transformer scale? (Phase 1: pi dominant for MLP)
  2. How many constants are needed for strong OOD generalization?
  3. Are any constants actively harmful (negative importance)?
  4. What is the minimal effective constant set?

Setup mirrors Phase 6 exactly (same architecture, data, hyperparams) but
varies only the SPIRNOR constant set used in the numeric embedding.

Experiments (17 total):
  - 1 full baseline (all 8 constants)
  - 8 leave-one-out ablations (remove one constant each)
  - 8 cumulative additions (1 constant, then 2, ..., up to 8)

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
SPIRNOR_CONST_NAMES = list(SPIRNOR_CONSTANTS.keys())

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
# NUMERIC EMBEDDING (supports variable constant sets)
# ============================================================

class SPIRNORNumericEmbedding(nn.Module):
    """SPIRNOR embedding with configurable constant subset."""
    def __init__(self, d_model, const_values, const_names=None):
        super().__init__()
        self.register_buffer('const_vals',
                             torch.tensor(const_values, dtype=torch.float32))
        self.n_consts = len(const_values)
        self.const_names = const_names or [f'c{i}' for i in range(self.n_consts)]
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
            theta = (C * n_vals) % (2 * math.pi)
            phi_angle = (phi_val * n_vals) % (2 * math.pi)
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


# ============================================================
# ROTARY POSITION ENCODING (Standard RoPE only for Phase 7)
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
    def __init__(self, d_model, nhead, num_layers, const_values, const_names,
                 task_configs, max_len=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_embed = SPIRNORNumericEmbedding(d_model, const_values, const_names)
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

        if (epoch + 1) in [1, 10, 20, 30]:
            print(f"      Epoch {epoch + 1:3d}/{epochs}: "
                  f"loss={avg_loss:.4f}, acc={avg_acc:.3f}")

    return n_params


# ============================================================
# EVALUATION
# ============================================================

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
# SINGLE ABLATION RUN
# ============================================================

def run_single_config(config_name, const_values, const_names,
                      train_data, test_datasets,
                      d_model, nhead, num_layers, epochs, batch_size, lr):
    """Train and evaluate one SPIRNOR configuration."""

    print(f"\n  {'-' * 56}")
    print(f"  CONFIG: {config_name}")
    print(f"  Constants ({len(const_names)}): {', '.join(const_names)}")
    print(f"  Raw features: {7 * len(const_names)} -> d_model={d_model}")
    print(f"  {'-' * 56}")

    task_configs = {name: cfg for name, cfg in TASKS.items()}
    model = NumericTransformer(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        const_values=const_values,
        const_names=const_names,
        task_configs=task_configs,
        max_len=4,
        dropout=0.1,
    )

    train_task_ids, train_numbers, train_labels = train_data

    t0 = time.time()
    n_params = train_model(model, train_task_ids, train_numbers, train_labels,
                           epochs=epochs, batch_size=batch_size, lr=lr)
    train_time = time.time() - t0
    print(f"      Training: {train_time:.1f}s, Params: {n_params:,}")

    config_results = OrderedDict()
    config_results['params'] = n_params
    config_results['train_time'] = round(train_time, 1)
    config_results['n_constants'] = len(const_names)
    config_results['constants'] = const_names

    range_keys = list(test_datasets.keys())
    print(f"      Evaluation:")
    for range_name, data in test_datasets.items():
        t_ids, nums, lbls = data
        results = evaluate(model, t_ids, nums, lbls)
        config_results[range_name] = results

        task_str = ' '.join(
            f"{TASKS[t]['name']}={results[t]:.3f}" for t in TASK_LIST
        )
        print(f"        {range_name:18s}: overall={results['overall']:.4f}  "
              f"[{task_str}]")

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return config_results


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment():
    print("=" * 70)
    print("SPIRNOR AI PHASE 7: CONSTANT ABLATION AT TRANSFORMER SCALE")
    print("=" * 70)
    print()

    # Hyperparameters — identical to Phase 6
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
    print(f"Precomputing SPF sieve up to {max_eval_num:,}...")
    t0 = time.time()
    spf_sieve = sieve_spf(max_eval_num)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Generate training data (same seed as Phase 6)
    print(f"\nGenerating training data (range {train_range})...")
    t0 = time.time()
    rng = np.random.RandomState(42)
    train_data = generate_dataset(n_train_per_task, train_range, rng, spf_sieve)
    print(f"  {len(train_data[0]):,} examples in {time.time() - t0:.1f}s")

    # Generate test data (same seeds as Phase 6)
    print(f"Generating test data...")
    test_datasets = OrderedDict()
    for range_name, r in test_ranges.items():
        test_rng = np.random.RandomState(hash(range_name) % 2**31)
        data = generate_dataset(n_eval_per_task, r, test_rng, spf_sieve)
        test_datasets[range_name] = data
        print(f"  {range_name:18s}: {len(data[0]):,} examples (range {r[0]:,}-{r[1]:,})")

    all_results = OrderedDict()

    # ============================================================
    # STUDY A: FULL BASELINE (all 8 constants)
    # ============================================================

    print(f"\n{'=' * 70}")
    print(f"STUDY A: FULL BASELINE (all 8 constants)")
    print(f"{'=' * 70}")

    full_vals = list(SPIRNOR_CONSTANTS.values())
    full_names = list(SPIRNOR_CONSTANTS.keys())

    res = run_single_config(
        'Full (8 constants)', full_vals, full_names,
        train_data, test_datasets,
        d_model, nhead, num_layers, epochs, batch_size, lr
    )
    all_results['full_8'] = res
    full_results = res  # Reference for importance computation

    # ============================================================
    # STUDY B: LEAVE-ONE-OUT ABLATION
    # ============================================================

    print(f"\n{'=' * 70}")
    print(f"STUDY B: LEAVE-ONE-OUT ABLATION (remove one constant at a time)")
    print(f"{'=' * 70}")

    loo_results = OrderedDict()

    for remove_idx, remove_name in enumerate(SPIRNOR_CONST_NAMES):
        remaining_vals = [v for i, v in enumerate(full_vals) if i != remove_idx]
        remaining_names = [n for i, n in enumerate(full_names) if i != remove_idx]

        config_name = f'Without {remove_name} (7 constants)'
        res = run_single_config(
            config_name, remaining_vals, remaining_names,
            train_data, test_datasets,
            d_model, nhead, num_layers, epochs, batch_size, lr
        )
        all_results[f'loo_{remove_name}'] = res
        loo_results[remove_name] = res

    # ============================================================
    # STUDY C: CUMULATIVE ADDITION (add constants one by one)
    # ============================================================

    print(f"\n{'=' * 70}")
    print(f"STUDY C: CUMULATIVE ADDITION (pi first, then add one at a time)")
    print(f"{'=' * 70}")

    cumulative_results = OrderedDict()

    for n_consts in range(1, len(full_names) + 1):
        subset_vals = full_vals[:n_consts]
        subset_names = full_names[:n_consts]

        config_name = f'{n_consts} constant{"s" if n_consts > 1 else ""}: {"+".join(subset_names)}'
        res = run_single_config(
            config_name, subset_vals, subset_names,
            train_data, test_datasets,
            d_model, nhead, num_layers, epochs, batch_size, lr
        )
        all_results[f'cumul_{n_consts}'] = res
        cumulative_results[n_consts] = res

    return all_results, full_results, loo_results, cumulative_results


# ============================================================
# RESULTS SUMMARY
# ============================================================

def print_summary(all_results, full_results, loo_results, cumulative_results):
    range_keys = ['in_range', 'ood_2k_5k', 'ood_5k_20k',
                  'ood_20k_100k', 'ood_100k_500k']
    range_labels = ['In-Range', '2K-5K', '5K-20K', '20K-100K', '100K-500K']

    print("\n" + "=" * 70)
    print("PHASE 7 RESULTS SUMMARY")
    print("=" * 70)

    # -- LEAVE-ONE-OUT IMPORTANCE --

    print("\n" + "-" * 70)
    print("LEAVE-ONE-OUT IMPORTANCE SCORES")
    print("(positive = constant helps, negative = constant hurts)")
    print("-" * 70)

    # Overall importance
    print("\n  Overall accuracy importance:")
    importance_overall = {}
    for const_name, loo_res in loo_results.items():
        scores = {}
        for key, label in zip(range_keys, range_labels):
            full_acc = full_results.get(key, {}).get('overall', 0)
            loo_acc = loo_res.get(key, {}).get('overall', 0)
            importance = full_acc - loo_acc  # positive = removing it hurts = it helps
            scores[label] = importance
        importance_overall[const_name] = scores

    # Print header
    header = f"  {'Constant':15s}"
    for label in range_labels:
        header += f" | {label:>9s}"
    header += f" | {'Avg OOD':>9s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Sort by average OOD importance
    avg_ood = {}
    for const_name, scores in importance_overall.items():
        ood_vals = [scores[l] for l in range_labels[1:]]  # exclude in-range
        avg_ood[const_name] = sum(ood_vals) / len(ood_vals)

    for const_name in sorted(avg_ood, key=lambda x: -avg_ood[x]):
        scores = importance_overall[const_name]
        line = f"  {const_name:15s}"
        for label in range_labels:
            val = scores[label]
            line += f" | {val:+.4f}  "
        line += f" | {avg_ood[const_name]:+.4f}  "
        print(line)

    # Per-task importance at 2K-5K (the primary OOD range)
    print("\n\n  Per-task importance at 2K-5K OOD:")
    task_importance = {}
    for const_name, loo_res in loo_results.items():
        for task in TASK_LIST:
            full_acc = full_results.get('ood_2k_5k', {}).get(task, 0)
            loo_acc = loo_res.get('ood_2k_5k', {}).get(task, 0)
            if const_name not in task_importance:
                task_importance[const_name] = {}
            task_importance[const_name][task] = full_acc - loo_acc

    header = f"  {'Constant':15s}"
    for task in TASK_LIST:
        header += f" | {TASKS[task]['name']:>8s}"
    header += f" | {'Avg':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for const_name in sorted(avg_ood, key=lambda x: -avg_ood[x]):
        line = f"  {const_name:15s}"
        vals = []
        for task in TASK_LIST:
            val = task_importance[const_name][task]
            vals.append(val)
            line += f" | {val:+.4f} "
        avg = sum(vals) / len(vals)
        line += f" | {avg:+.4f} "
        print(line)

    # -- CUMULATIVE ADDITION CURVES --

    print("\n\n" + "-" * 70)
    print("CUMULATIVE ADDITION CURVES")
    print("(adding constants one by one in order: pi, sqrt2, phi², e, ...)")
    print("-" * 70)

    print("\n  Overall accuracy by number of constants:")
    header = f"  {'N':3s} {'Constants':40s}"
    for label in range_labels:
        header += f" | {label:>9s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for n_consts, cum_res in cumulative_results.items():
        names_str = '+'.join(cum_res.get('constants', []))
        if len(names_str) > 40:
            names_str = names_str[:37] + '...'
        line = f"  {n_consts:3d} {names_str:40s}"
        for key in range_keys:
            val = cum_res.get(key, {}).get('overall', 0)
            line += f" | {val:.4f}   "
        print(line)

    # Marginal gains per added constant
    print("\n  Marginal OOD gain per added constant (avg over 4 OOD ranges):")
    prev_ood = None
    for n_consts, cum_res in cumulative_results.items():
        ood_vals = [cum_res.get(key, {}).get('overall', 0) for key in range_keys[1:]]
        avg_ood_acc = sum(ood_vals) / len(ood_vals)
        if prev_ood is not None:
            gain = avg_ood_acc - prev_ood
            added = cum_res.get('constants', ['?'] * n_consts)[n_consts - 1]
            bar = '#' * max(0, int(gain * 200))
            print(f"    +{added:15s} (->  {n_consts} const): "
                  f"avg_OOD={avg_ood_acc:.4f} (gain={gain:+.4f}) {bar}")
        else:
            added = cum_res.get('constants', ['?'])[0]
            print(f"    {added:16s} (   1 const): "
                  f"avg_OOD={avg_ood_acc:.4f}")
        prev_ood = avg_ood_acc

    # -- ESSENTIAL SET ANALYSIS --

    print("\n\n" + "-" * 70)
    print("ESSENTIAL SET ANALYSIS")
    print("-" * 70)

    # Find minimum N where cumulative OOD performance reaches 95% of full
    full_ood_vals = [full_results.get(key, {}).get('overall', 0)
                     for key in range_keys[1:]]
    full_avg_ood = sum(full_ood_vals) / len(full_ood_vals)

    print(f"\n  Full 8-constant avg OOD accuracy: {full_avg_ood:.4f}")
    print(f"  95% threshold: {full_avg_ood * 0.95:.4f}")
    print(f"  99% threshold: {full_avg_ood * 0.99:.4f}")

    for threshold_pct, threshold_name in [(95, '95%'), (99, '99%')]:
        threshold = full_avg_ood * threshold_pct / 100
        found = False
        for n_consts, cum_res in cumulative_results.items():
            ood_vals = [cum_res.get(key, {}).get('overall', 0)
                        for key in range_keys[1:]]
            avg = sum(ood_vals) / len(ood_vals)
            if avg >= threshold:
                names = '+'.join(cum_res.get('constants', []))
                print(f"  {threshold_name} reached at N={n_consts}: "
                      f"avg_OOD={avg:.4f} ({names})")
                found = True
                break
        if not found:
            print(f"  {threshold_name} NOT reached even with all 8 constants")

    # -- PARAMETER EFFICIENCY --

    print("\n\n" + "-" * 70)
    print("PARAMETER COUNTS")
    print("-" * 70)

    for n_consts, cum_res in cumulative_results.items():
        names = '+'.join(cum_res.get('constants', []))
        params = cum_res.get('params', 0)
        print(f"  {n_consts} constants: {params:,} params  ({names})")


def save_results(all_results, full_results, loo_results, cumulative_results):
    """Save all results to JSON."""

    range_keys = ['in_range', 'ood_2k_5k', 'ood_5k_20k',
                  'ood_20k_100k', 'ood_100k_500k']

    # Compute importance scores
    importance = {}
    for const_name, loo_res in loo_results.items():
        importance[const_name] = {}
        for key in range_keys:
            importance[const_name][key] = {}
            for task in TASK_LIST + ['overall']:
                full_acc = full_results.get(key, {}).get(task, 0)
                loo_acc = loo_res.get(key, {}).get(task, 0)
                importance[const_name][key][task] = round(full_acc - loo_acc, 4)

        # Average OOD importance
        ood_overall = [importance[const_name][k]['overall'] for k in range_keys[1:]]
        importance[const_name]['avg_ood_overall'] = round(
            sum(ood_overall) / len(ood_overall), 4)

    def clean(obj):
        if isinstance(obj, (OrderedDict, dict)):
            return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    save_path = 'Scripts/SPIRNOR_AI_PHASE7_RESULTS.json'
    save_data = {
        'phase': 'Phase 7: Constant Ablation at Transformer Scale',
        'config': {
            'd_model': 128, 'nhead': 4, 'num_layers': 4, 'epochs': 30,
            'train_range': [2, 2000],
            'n_train_per_task': 50000,
            'n_eval_per_task': 1000,
            'spirnor_constants': dict(SPIRNOR_CONSTANTS),
            'constant_order': list(SPIRNOR_CONSTANTS.keys()),
        },
        'results': clean(all_results),
        'importance_scores': clean(importance),
    }

    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {save_path}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    all_results, full_results, loo_results, cumulative_results = run_experiment()
    print_summary(all_results, full_results, loo_results, cumulative_results)
    save_results(all_results, full_results, loo_results, cumulative_results)
