#!/usr/bin/env python3
"""
SPIRNOR AI Phase 7B: Multi-Seed Statistical Validation

Validates key Phase 7 findings across 5 random seeds to provide
confidence intervals and confirm that results are not seed artifacts.

Key claims to validate:
  1. Pi alone (1 const) gives BEST average OOD — near-zero degradation
  2. 3 constants (pi+sqrt2+phi_sq) is the sweet spot for in-range + OOD
  3. Full 8 constants overfits relative to smaller subsets
  4. SPIRNOR consistently beats Learned embedding OOD

Configurations tested (5 seeds each = 25 runs total):
  A. Learned+RoPE             (Phase 6 baseline reference)
  B. SPIRNOR 1 const (pi)     (Phase 7 surprise — best OOD?)
  C. SPIRNOR 3 const          (Phase 7 sweet spot)
  D. SPIRNOR 8 const (full)   (Phase 6 winner)
  E. SPIRNOR 7 const (no pi)  (confirms pi dominance)

Seeds: 42, 123, 456, 789, 2024
Training data varies per seed. Test data is FIXED across seeds.

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

# Constant subsets
CONST_SETS = {
    '1_pi': (['pi'], [math.pi]),
    '3_sweet': (['pi', 'sqrt2', 'phi_sq'], [math.pi, math.sqrt(2), PHI ** 2]),
    '7_no_pi': (SPIRNOR_CONST_NAMES[1:], SPIRNOR_CONST_LIST[1:]),
    '8_full': (SPIRNOR_CONST_NAMES, SPIRNOR_CONST_LIST),
}

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
# EMBEDDINGS
# ============================================================

class SPIRNORNumericEmbedding(nn.Module):
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


class LearnedNumericEmbedding(nn.Module):
    def __init__(self, max_num, d_model):
        super().__init__()
        self.max_num = max_num
        self.embed = nn.Embedding(max_num + 1, d_model)

    def forward(self, numbers):
        idx = numbers.long() % (self.max_num + 1)
        return self.embed(idx)

# ============================================================
# ROTARY PE
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
# TRANSFORMER
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
                 const_values=None, const_names=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        if embed_type == 'learned':
            self.num_embed = LearnedNumericEmbedding(max_num, d_model)
        elif embed_type == 'spirnor':
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
    all_task_ids, all_numbers, all_labels = [], [], []

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
        total_loss, total_correct, total_samples = 0.0, 0, 0

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
                task_loss = F.cross_entropy(task_logits, task_labels, reduction='sum')
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

        if (epoch + 1) in [1, 15, 30]:
            avg_loss = total_loss / max(total_samples, 1)
            avg_acc = total_correct / max(total_samples, 1)
            print(f"        Ep {epoch+1:2d}: loss={avg_loss:.4f} acc={avg_acc:.3f}")

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
# MAIN EXPERIMENT
# ============================================================

def run_experiment():
    print("=" * 70)
    print("SPIRNOR AI PHASE 7B: MULTI-SEED STATISTICAL VALIDATION")
    print("=" * 70)

    # Hyperparameters -- identical to Phase 6/7
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

    seeds = [42, 123, 456, 789, 2024]

    configs = OrderedDict([
        ('Learned+RoPE', {'embed': 'learned', 'const_set': None}),
        ('SPIRNOR_1_pi', {'embed': 'spirnor', 'const_set': '1_pi'}),
        ('SPIRNOR_3_sweet', {'embed': 'spirnor', 'const_set': '3_sweet'}),
        ('SPIRNOR_8_full', {'embed': 'spirnor', 'const_set': '8_full'}),
        ('SPIRNOR_7_no_pi', {'embed': 'spirnor', 'const_set': '7_no_pi'}),
    ])

    task_configs = {name: cfg for name, cfg in TASKS.items()}

    # Precompute SPF sieve
    max_eval_num = max(hi for _, hi in test_ranges.values())
    print(f"\nPrecomputing SPF sieve up to {max_eval_num:,}...")
    t0 = time.time()
    spf_sieve = sieve_spf(max_eval_num)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Generate FIXED test data (same across all seeds)
    print(f"Generating fixed test data...")
    test_datasets = OrderedDict()
    for range_name, r in test_ranges.items():
        test_rng = np.random.RandomState(hash(range_name) % 2**31)
        data = generate_dataset(n_eval_per_task, r, test_rng, spf_sieve)
        test_datasets[range_name] = data
        print(f"  {range_name:18s}: {len(data[0]):,} examples")

    # Results storage: config -> list of per-seed results
    all_results = OrderedDict()
    for config_name in configs:
        all_results[config_name] = []

    total_runs = len(configs) * len(seeds)
    run_num = 0

    print(f"\n{'=' * 70}")
    print(f"RUNNING {total_runs} EXPERIMENTS ({len(configs)} configs x {len(seeds)} seeds)")
    print(f"{'=' * 70}")

    for config_name, cfg in configs.items():
        print(f"\n{'=' * 60}")
        print(f"CONFIG: {config_name}")
        print(f"{'=' * 60}")

        for seed in seeds:
            run_num += 1
            print(f"\n  --- Seed {seed} (run {run_num}/{total_runs}) ---")

            # Set seeds for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            # Generate training data with this seed
            train_rng = np.random.RandomState(seed)
            t0 = time.time()
            train_data = generate_dataset(n_train_per_task, train_range,
                                          train_rng, spf_sieve)
            gen_time = time.time() - t0
            print(f"      Data: {len(train_data[0]):,} examples ({gen_time:.1f}s)")

            # Build model
            if cfg['embed'] == 'learned':
                model = NumericTransformer(
                    d_model=d_model, nhead=nhead, num_layers=num_layers,
                    embed_type='learned', task_configs=task_configs,
                    max_num=train_range[1], max_len=4, dropout=0.1,
                )
            else:
                cset = cfg['const_set']
                const_names, const_values = CONST_SETS[cset]
                model = NumericTransformer(
                    d_model=d_model, nhead=nhead, num_layers=num_layers,
                    embed_type='spirnor', task_configs=task_configs,
                    max_num=train_range[1], max_len=4,
                    const_values=const_values, const_names=const_names,
                    dropout=0.1,
                )

            # Train
            t0 = time.time()
            n_params = train_model(model, *train_data,
                                   epochs=epochs, batch_size=batch_size, lr=lr)
            train_time = time.time() - t0
            print(f"      Train: {train_time:.1f}s, {n_params:,} params")

            # Evaluate on all fixed test ranges
            seed_results = OrderedDict()
            seed_results['seed'] = seed
            seed_results['params'] = n_params
            seed_results['train_time'] = round(train_time, 1)

            for range_name, data in test_datasets.items():
                results = evaluate(model, *data)
                seed_results[range_name] = results

            # Print compact summary
            range_keys = list(test_ranges.keys())
            overall_str = '  '.join(
                f"{k.split('_', 1)[-1] if '_' in k else k}="
                f"{seed_results[k]['overall']:.3f}"
                for k in range_keys
            )
            print(f"      Results: {overall_str}")

            all_results[config_name].append(seed_results)

            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    return all_results, list(test_ranges.keys())


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

def compute_stats(values):
    """Compute mean, std, min, max for a list of values."""
    arr = np.array(values)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'n': len(arr),
    }


def welch_t_test(vals_a, vals_b):
    """Two-sample Welch's t-test. Returns t-statistic and approximate p-value."""
    a, b = np.array(vals_a), np.array(vals_b)
    n_a, n_b = len(a), len(b)
    mean_a, mean_b = np.mean(a), np.mean(b)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)

    se = np.sqrt(var_a / n_a + var_b / n_b)
    if se < 1e-10:
        return 0.0, 1.0

    t_stat = (mean_a - mean_b) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = num / max(denom, 1e-10)

    # Approximate p-value using normal distribution (good for df > 4)
    from math import erfc
    p_value = erfc(abs(t_stat) / math.sqrt(2))

    return float(t_stat), float(p_value)


def print_summary(all_results, range_keys):
    range_labels = {
        'in_range': 'In-Range',
        'ood_2k_5k': '2K-5K',
        'ood_5k_20k': '5K-20K',
        'ood_20k_100k': '20K-100K',
        'ood_100k_500k': '100K-500K',
    }

    print("\n" + "=" * 70)
    print("PHASE 7B: MULTI-SEED STATISTICAL SUMMARY")
    print("=" * 70)

    # ---- Overall accuracy: mean +/- std ----

    print("\n" + "-" * 70)
    print("OVERALL ACCURACY: mean +/- std (5 seeds)")
    print("-" * 70)

    header = f"  {'Config':20s}"
    for key in range_keys:
        header += f" | {range_labels[key]:>14s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    config_stats = OrderedDict()
    for config_name, seed_runs in all_results.items():
        stats = OrderedDict()
        for key in range_keys:
            vals = [run[key]['overall'] for run in seed_runs]
            stats[key] = compute_stats(vals)
        config_stats[config_name] = stats

        line = f"  {config_name:20s}"
        for key in range_keys:
            s = stats[key]
            line += f" | {s['mean']:.3f}+/-{s['std']:.3f}"
        print(line)

    # ---- Average OOD (mean across 4 OOD ranges) ----

    print("\n" + "-" * 70)
    print("AVERAGE OOD ACCURACY (mean of 4 OOD ranges, per seed)")
    print("-" * 70)

    ood_keys = range_keys[1:]
    avg_ood_per_config = OrderedDict()

    for config_name, seed_runs in all_results.items():
        per_seed_avg_ood = []
        for run in seed_runs:
            ood_vals = [run[k]['overall'] for k in ood_keys]
            per_seed_avg_ood.append(np.mean(ood_vals))
        avg_ood_per_config[config_name] = per_seed_avg_ood
        stats = compute_stats(per_seed_avg_ood)
        seeds_str = ', '.join(f"{v:.3f}" for v in per_seed_avg_ood)
        print(f"  {config_name:20s}: {stats['mean']:.4f} +/- {stats['std']:.4f}  "
              f"[{seeds_str}]")

    # ---- Statistical tests ----

    print("\n" + "-" * 70)
    print("STATISTICAL COMPARISONS (Welch's t-test)")
    print("-" * 70)

    comparisons = [
        ('SPIRNOR_8_full', 'Learned+RoPE', 'SPIRNOR(8) vs Learned'),
        ('SPIRNOR_3_sweet', 'Learned+RoPE', 'SPIRNOR(3) vs Learned'),
        ('SPIRNOR_1_pi', 'Learned+RoPE', 'SPIRNOR(1) vs Learned'),
        ('SPIRNOR_1_pi', 'SPIRNOR_8_full', 'Pi-only vs Full(8)'),
        ('SPIRNOR_3_sweet', 'SPIRNOR_8_full', 'Sweet(3) vs Full(8)'),
        ('SPIRNOR_8_full', 'SPIRNOR_7_no_pi', 'Full(8) vs No-Pi(7)'),
    ]

    for name_a, name_b, label in comparisons:
        if name_a not in avg_ood_per_config or name_b not in avg_ood_per_config:
            continue

        print(f"\n  {label}:")

        # Test at each range
        for key in range_keys:
            vals_a = [run[key]['overall'] for run in all_results[name_a]]
            vals_b = [run[key]['overall'] for run in all_results[name_b]]
            t_stat, p_val = welch_t_test(vals_a, vals_b)
            mean_a, mean_b = np.mean(vals_a), np.mean(vals_b)
            diff = mean_a - mean_b
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"    {range_labels[key]:>10s}: {mean_a:.4f} vs {mean_b:.4f}  "
                  f"diff={diff:+.4f}  t={t_stat:+.2f}  p={p_val:.4f} {sig}")

        # Test on avg OOD
        vals_a = avg_ood_per_config[name_a]
        vals_b = avg_ood_per_config[name_b]
        t_stat, p_val = welch_t_test(vals_a, vals_b)
        diff = np.mean(vals_a) - np.mean(vals_b)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"    {'Avg OOD':>10s}: {np.mean(vals_a):.4f} vs {np.mean(vals_b):.4f}  "
              f"diff={diff:+.4f}  t={t_stat:+.2f}  p={p_val:.4f} {sig}")

    # ---- Per-task breakdown at 2K-5K ----

    print("\n" + "-" * 70)
    print("PER-TASK ACCURACY AT 2K-5K: mean +/- std")
    print("-" * 70)

    for task in TASK_LIST:
        task_label = TASKS[task]['name']
        print(f"\n  {task_label}:")
        for config_name, seed_runs in all_results.items():
            vals = [run['ood_2k_5k'][task] for run in seed_runs]
            stats = compute_stats(vals)
            print(f"    {config_name:20s}: {stats['mean']:.3f} +/- {stats['std']:.3f}  "
                  f"[{stats['min']:.3f} - {stats['max']:.3f}]")

    # ---- Key claims validation ----

    print("\n" + "-" * 70)
    print("KEY CLAIMS VALIDATION")
    print("-" * 70)

    # Claim 1: Pi alone gives best avg OOD
    pi_ood = avg_ood_per_config.get('SPIRNOR_1_pi', [])
    full_ood = avg_ood_per_config.get('SPIRNOR_8_full', [])
    if pi_ood and full_ood:
        pi_mean = np.mean(pi_ood)
        full_mean = np.mean(full_ood)
        t, p = welch_t_test(pi_ood, full_ood)
        verdict = "CONFIRMED" if pi_mean > full_mean and p < 0.05 else \
                  "TREND" if pi_mean > full_mean else "REJECTED"
        print(f"\n  Claim 1: Pi alone gives best avg OOD")
        print(f"    Pi(1): {pi_mean:.4f} vs Full(8): {full_mean:.4f}  "
              f"diff={pi_mean - full_mean:+.4f}  p={p:.4f}  -> {verdict}")

    # Claim 2: 3 constants is the sweet spot
    sweet_ood = avg_ood_per_config.get('SPIRNOR_3_sweet', [])
    if sweet_ood and full_ood:
        sweet_mean = np.mean(sweet_ood)
        t, p = welch_t_test(sweet_ood, full_ood)
        verdict = "CONFIRMED" if sweet_mean > full_mean and p < 0.05 else \
                  "TREND" if sweet_mean > full_mean else "REJECTED"
        print(f"\n  Claim 2: 3 constants better OOD than 8")
        print(f"    Sweet(3): {sweet_mean:.4f} vs Full(8): {full_mean:.4f}  "
              f"diff={sweet_mean - full_mean:+.4f}  p={p:.4f}  -> {verdict}")

    # Claim 3: Removing pi is catastrophic
    nopi_ood = avg_ood_per_config.get('SPIRNOR_7_no_pi', [])
    if nopi_ood and full_ood:
        nopi_mean = np.mean(nopi_ood)
        t, p = welch_t_test(full_ood, nopi_ood)
        verdict = "CONFIRMED" if full_mean > nopi_mean and p < 0.05 else \
                  "TREND" if full_mean > nopi_mean else "REJECTED"
        print(f"\n  Claim 3: Removing pi is catastrophic")
        print(f"    Full(8): {full_mean:.4f} vs No-Pi(7): {nopi_mean:.4f}  "
              f"diff={full_mean - nopi_mean:+.4f}  p={p:.4f}  -> {verdict}")

    # Claim 4: SPIRNOR beats Learned OOD
    learned_ood = avg_ood_per_config.get('Learned+RoPE', [])
    if full_ood and learned_ood:
        learned_mean = np.mean(learned_ood)
        t, p = welch_t_test(full_ood, learned_ood)
        verdict = "CONFIRMED" if full_mean > learned_mean and p < 0.05 else \
                  "TREND" if full_mean > learned_mean else "REJECTED"
        print(f"\n  Claim 4: SPIRNOR(8) beats Learned OOD")
        print(f"    SPIRNOR(8): {full_mean:.4f} vs Learned: {learned_mean:.4f}  "
              f"diff={full_mean - learned_mean:+.4f}  p={p:.4f}  -> {verdict}")

    return config_stats, avg_ood_per_config


def save_results(all_results, range_keys, config_stats, avg_ood_per_config):
    def clean(obj):
        if isinstance(obj, (OrderedDict, dict)):
            return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        return obj

    save_data = {
        'phase': 'Phase 7B: Multi-Seed Statistical Validation',
        'seeds': [42, 123, 456, 789, 2024],
        'config': {
            'd_model': 128, 'nhead': 4, 'num_layers': 4, 'epochs': 30,
            'train_range': [2, 2000],
            'n_train_per_task': 50000,
            'n_eval_per_task': 1000,
        },
        'raw_results': clean(all_results),
        'summary_stats': clean(config_stats),
        'avg_ood_per_config': clean(avg_ood_per_config),
    }

    save_path = 'Scripts/SPIRNOR_AI_PHASE7B_RESULTS.json'
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {save_path}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    all_results, range_keys = run_experiment()
    config_stats, avg_ood = print_summary(all_results, range_keys)
    save_results(all_results, range_keys, config_stats, avg_ood)
