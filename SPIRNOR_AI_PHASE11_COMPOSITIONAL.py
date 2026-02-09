#!/usr/bin/env python3
"""
SPIRNOR AI Phase 11: Compositional & Structured Numeric Reasoning

Building on Phase 8's rational constants (C=2pi/p) and Phase 9's task expansion,
this phase pushes SPIRNOR into genuinely novel territory:
  - Compositional modular reasoning (multi-step, multi-number)
  - Variable-length set aggregation (3-8 elements)
  - Sequence pattern detection

8 Tasks across 4 Families:
  A. CRT Validation:     Mod30(n), ProductMod30(a,b)
  B. Compositional:      SumMod30(a,b,c), DiophantineSolvable(a,b,c), GCDEquals(a,b,c)
  C. Set Reasoning:      SetAllDivisible(S,p), SetCoprime(S)
  D. Pattern Detection:  HiddenModulus(seq)

2 embedding configs: SPIRNOR rational_5 vs Learned baseline
Training: 400K examples (50K per task x 8 tasks), train 2-2000, 30 epochs
Eval ranges: in_range, 2K-5K, 5K-20K, 20K-100K, 100K-500K

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

MAX_SEQ = 8  # Maximum numbers per example

# ============================================================
# TASK DEFINITIONS (8 tasks across 4 families)
# ============================================================

TASKS = OrderedDict([
    # Family A: CRT Validation
    ('mod30',        {'n_inputs': 1, 'n_classes': 30, 'name': 'Mod30',       'family': 'A'}),
    ('prodmod30',    {'n_inputs': 2, 'n_classes': 30, 'name': 'ProdMod30',   'family': 'A'}),
    # Family B: Compositional Multi-Step
    ('summod30',     {'n_inputs': 3, 'n_classes': 30, 'name': 'SumMod30',    'family': 'B'}),
    ('diophantine',  {'n_inputs': 3, 'n_classes': 2,  'name': 'Diophantine', 'family': 'B'}),
    ('gcdequals',    {'n_inputs': 3, 'n_classes': 2,  'name': 'GCDEquals',   'family': 'B'}),
    # Family C: Variable-Length Set
    ('setalldiv',    {'n_inputs': 8, 'n_classes': 2,  'name': 'SetAllDiv',   'family': 'C'}),
    ('setcoprime',   {'n_inputs': 8, 'n_classes': 2,  'name': 'SetCoprime',  'family': 'C'}),
    # Family D: Sequence Pattern
    ('hiddenmod',    {'n_inputs': 8, 'n_classes': 5,  'name': 'HiddenMod',   'family': 'D'}),
])

TASK_LIST = list(TASKS.keys())
TASK_ID = {name: i for i, name in enumerate(TASK_LIST)}
N_TASKS = len(TASK_LIST)

# ============================================================
# NUMBER THEORY HELPERS
# ============================================================

def multi_gcd(numbers):
    """GCD of a list of numbers."""
    result = numbers[0]
    for n in numbers[1:]:
        result = gcd(result, n)
        if result == 1:
            return 1
    return result


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
                 task_configs, max_num=2000, max_len=10,
                 dropout=0.1, const_values=None, const_names=None):
        super().__init__()
        self.d_model = d_model
        self.embed_type = embed_type

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

def generate_dataset(n_per_task, num_range, rng):
    """Generate training/eval data for all 8 Phase 11 tasks."""
    lo, hi = num_range
    all_task_ids = []
    all_numbers = []
    all_labels = []

    for task_name in TASK_LIST:
        task_id = TASK_ID[task_name]

        if task_name == 'mod30':
            # n mod 30 (30 classes)
            for _ in range(n_per_task):
                a = int(rng.randint(lo, hi + 1))
                nums = [a] + [0] * (MAX_SEQ - 1)
                label = a % 30
                all_task_ids.append(task_id)
                all_numbers.append(nums)
                all_labels.append(label)

        elif task_name == 'prodmod30':
            # (a * b) mod 30 (30 classes)
            for _ in range(n_per_task):
                a = int(rng.randint(lo, hi + 1))
                b = int(rng.randint(lo, hi + 1))
                nums = [a, b] + [0] * (MAX_SEQ - 2)
                label = (a * b) % 30
                all_task_ids.append(task_id)
                all_numbers.append(nums)
                all_labels.append(label)

        elif task_name == 'summod30':
            # (a + b + c) mod 30 (30 classes)
            for _ in range(n_per_task):
                a = int(rng.randint(lo, hi + 1))
                b = int(rng.randint(lo, hi + 1))
                c = int(rng.randint(lo, hi + 1))
                nums = [a, b, c] + [0] * (MAX_SEQ - 3)
                label = (a + b + c) % 30
                all_task_ids.append(task_id)
                all_numbers.append(nums)
                all_labels.append(label)

        elif task_name == 'diophantine':
            # ax + by = c solvable? (binary) Requires gcd(a,b) | c
            n_each = n_per_task // 2

            # Positive: gcd(a,b) divides c
            count = 0
            while count < n_each:
                a = int(rng.randint(lo, hi + 1))
                b = int(rng.randint(lo, hi + 1))
                g = gcd(a, b)
                max_k = hi // g
                if max_k < 1:
                    continue
                k = int(rng.randint(1, max_k + 1))
                c = g * k
                if c < lo or c > hi:
                    c = max(lo, min(c, hi))
                    # Verify it's still valid
                    if c % g != 0:
                        continue
                nums = [a, b, c] + [0] * (MAX_SEQ - 3)
                all_task_ids.append(task_id)
                all_numbers.append(nums)
                all_labels.append(1)
                count += 1

            # Negative: gcd(a,b) does NOT divide c
            count = 0
            attempts = 0
            while count < n_per_task - n_each and attempts < n_per_task * 20:
                attempts += 1
                # Pick a shared prime factor to guarantee gcd > 1
                p = RATIONAL_5_PRIMES[int(rng.randint(0, len(RATIONAL_5_PRIMES)))]
                a_k = int(rng.randint(max((lo + p - 1) // p, 1), hi // p + 1))
                b_k = int(rng.randint(max((lo + p - 1) // p, 1), hi // p + 1))
                a = p * a_k
                b = p * b_k
                if a < lo or a > hi or b < lo or b > hi:
                    continue
                g = gcd(a, b)
                if g <= 1:
                    continue
                # c not divisible by g
                c = int(rng.randint(lo, hi + 1))
                if c % g == 0:
                    c += int(rng.randint(1, g))
                    if c > hi:
                        c = lo + (c % (hi - lo + 1))
                if c % g == 0:
                    continue  # still divisible, skip
                if c < lo or c > hi:
                    continue
                nums = [a, b, c] + [0] * (MAX_SEQ - 3)
                all_task_ids.append(task_id)
                all_numbers.append(nums)
                all_labels.append(0)
                count += 1

        elif task_name == 'gcdequals':
            # gcd(a,b) = c? (binary)
            n_each = n_per_task // 2

            # Positive: c = gcd(a,b)
            for _ in range(n_each):
                a = int(rng.randint(lo, hi + 1))
                b = int(rng.randint(lo, hi + 1))
                c = gcd(a, b)
                nums = [a, b, c] + [0] * (MAX_SEQ - 3)
                all_task_ids.append(task_id)
                all_numbers.append(nums)
                all_labels.append(1)

            # Negative: c != gcd(a,b), but sampled from similar range
            for _ in range(n_per_task - n_each):
                a = int(rng.randint(lo, hi + 1))
                b = int(rng.randint(lo, hi + 1))
                g = gcd(a, b)
                # Sample c from reasonable GCD-like range [1, 50] but != g
                candidates = [v for v in range(1, 51) if v != g]
                c = int(rng.choice(candidates))
                nums = [a, b, c] + [0] * (MAX_SEQ - 3)
                all_task_ids.append(task_id)
                all_numbers.append(nums)
                all_labels.append(0)

        elif task_name == 'setalldiv':
            # All elements divisible by prime p? (binary)
            # Input: [p, n1, n2, ..., nk, 0, 0, ...] where k=3..7
            n_each = n_per_task // 2

            # Positive: all divisible by p
            for _ in range(n_each):
                p = RATIONAL_5_PRIMES[int(rng.randint(0, len(RATIONAL_5_PRIMES)))]
                k = int(rng.randint(3, 8))  # set size 3-7
                max_mult = hi // p
                min_mult = max((lo + p - 1) // p, 1)  # ceiling division
                if min_mult > max_mult:
                    min_mult = 1
                elements = []
                for _ in range(k):
                    m = int(rng.randint(min_mult, max_mult + 1))
                    elements.append(p * m)
                rng.shuffle(elements)
                nums = [p] + list(elements) + [0] * (MAX_SEQ - 1 - k)
                all_task_ids.append(task_id)
                all_numbers.append(nums)
                all_labels.append(1)

            # Negative: at least one NOT divisible by p
            for _ in range(n_per_task - n_each):
                p = RATIONAL_5_PRIMES[int(rng.randint(0, len(RATIONAL_5_PRIMES)))]
                k = int(rng.randint(3, 8))
                max_mult = hi // p
                min_mult = max((lo + p - 1) // p, 1)  # ceiling division
                if min_mult > max_mult:
                    min_mult = 1
                elements = []
                # k-1 divisible
                for _ in range(k - 1):
                    m = int(rng.randint(min_mult, max_mult + 1))
                    elements.append(p * m)
                # 1 not divisible
                bad = int(rng.randint(lo, hi + 1))
                attempts = 0
                while bad % p == 0 and attempts < 100:
                    bad = int(rng.randint(lo, hi + 1))
                    attempts += 1
                elements.append(bad)
                rng.shuffle(elements)
                nums = [p] + list(elements) + [0] * (MAX_SEQ - 1 - k)
                all_task_ids.append(task_id)
                all_numbers.append(nums)
                all_labels.append(0)

        elif task_name == 'setcoprime':
            # GCD of entire set = 1? (binary)
            # Input: [n1, n2, ..., nk, 0, 0, ...] where k=3..8
            n_each = n_per_task // 2

            # Positive: GCD = 1 (rejection sampling)
            count = 0
            attempts = 0
            while count < n_each and attempts < n_per_task * 10:
                attempts += 1
                k = int(rng.randint(3, 9))  # set size 3-8
                elements = [int(rng.randint(lo, hi + 1)) for _ in range(k)]
                if multi_gcd(elements) == 1:
                    rng.shuffle(elements)
                    nums = list(elements) + [0] * (MAX_SEQ - k)
                    all_task_ids.append(task_id)
                    all_numbers.append(nums)
                    all_labels.append(1)
                    count += 1

            # Negative: GCD > 1 (all share a common factor)
            for _ in range(n_per_task - n_each):
                k = int(rng.randint(3, 9))
                # Pick a common divisor
                common_divisors = [2, 3, 4, 5, 6, 7, 10, 11, 13, 14, 15]
                d = int(rng.choice(common_divisors))
                max_mult = hi // d
                min_mult = max((lo + d - 1) // d, 1)  # ceiling division
                if min_mult > max_mult:
                    min_mult = 1
                elements = []
                for _ in range(k):
                    m = int(rng.randint(min_mult, max_mult + 1))
                    elements.append(d * m)
                rng.shuffle(elements)
                nums = list(elements) + [0] * (MAX_SEQ - k)
                all_task_ids.append(task_id)
                all_numbers.append(nums)
                all_labels.append(0)

        elif task_name == 'hiddenmod':
            # Identify which prime p in {2,3,5,7,11} creates shared residue
            # Input: 8 numbers all == r (mod p)
            for _ in range(n_per_task):
                generated = False
                gen_attempts = 0
                while not generated and gen_attempts < 100:
                    gen_attempts += 1
                    p_idx = int(rng.randint(0, 5))
                    p = RATIONAL_5_PRIMES[p_idx]
                    r = int(rng.randint(0, p))

                    # Generate 8 distinct numbers == r (mod p)
                    candidates = set()
                    inner_attempts = 0
                    while len(candidates) < 8 and inner_attempts < 500:
                        inner_attempts += 1
                        max_k = (hi - r) // p
                        min_k = max((lo - r + p - 1) // p, 0)
                        if r == 0:
                            min_k = max(min_k, 1)  # avoid n=0
                        if min_k > max_k:
                            break
                        kv = int(rng.randint(min_k, max_k + 1))
                        n_val = p * kv + r
                        if lo <= n_val <= hi and n_val > 0:
                            candidates.add(n_val)

                    if len(candidates) < 8:
                        continue

                    elements = list(candidates)[:8]

                    # Disambiguation: check no OTHER prime also has shared residue
                    ambiguous = False
                    for q_idx, q in enumerate(RATIONAL_5_PRIMES):
                        if q_idx == p_idx:
                            continue
                        residues = set(e % q for e in elements)
                        if len(residues) == 1:
                            ambiguous = True
                            break

                    if not ambiguous:
                        rng.shuffle(elements)
                        nums = list(elements)
                        all_task_ids.append(task_id)
                        all_numbers.append(nums)
                        all_labels.append(p_idx)
                        generated = True

                if not generated:
                    # Fallback: accept ambiguous, label with chosen p
                    p_idx = int(rng.randint(0, 5))
                    p = RATIONAL_5_PRIMES[p_idx]
                    r = int(rng.randint(0, p))
                    elements = []
                    for i in range(8):
                        max_k = (hi - r) // p
                        min_k = max((lo - r + p - 1) // p, 0)
                        if r == 0:
                            min_k = max(min_k, 1)
                        if min_k > max_k:
                            min_k = 0
                        kv = int(rng.randint(min_k, max(max_k, min_k) + 1))
                        n_val = p * kv + r
                        n_val = max(lo, min(n_val, hi))
                        elements.append(n_val)
                    rng.shuffle(elements)
                    nums = list(elements)
                    all_task_ids.append(task_id)
                    all_numbers.append(nums)
                    all_labels.append(p_idx)

    # Shuffle all examples
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
# EMBEDDING CONFIGS
# ============================================================

EMBED_CONFIGS = OrderedDict([
    ('spirnor_rational5', {
        'type': 'spirnor',
        'desc': 'Pure SPIRNOR with rational_5 {2pi/2,3,5,7,11}',
        'const_values': RATIONAL_5_VALUES,
        'const_names': RATIONAL_5_NAMES,
    }),
    ('learned', {
        'type': 'learned',
        'desc': 'Pure learned embedding (nn.Embedding baseline)',
        'const_values': None,
        'const_names': None,
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
    print(f"  d_model={d_model}, nhead={nhead}, num_layers={num_layers}")
    print(f"  {'-' * 56}")

    task_configs = {name: cfg for name, cfg in TASKS.items()}
    model = NumericTransformer(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        embed_type=config['type'],
        task_configs=task_configs,
        max_num=max_num,
        max_len=MAX_SEQ + 2,  # +1 for task token, +1 buffer
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
# PART A: MAIN EXPERIMENT (d_model=128, 2 configs, 8 tasks)
# ============================================================

def run_part_a():
    print("=" * 70)
    print("PHASE 11 PART A: COMPOSITIONAL REASONING (d_model=128)")
    print("2 embedding configs x 8 tasks")
    print("=" * 70)

    d_model = 128
    nhead = 4
    num_layers = 6
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

    # Generate training data (fixed seed)
    print(f"\nGenerating training data (range {train_range}, "
          f"{n_train_per_task} per task x {N_TASKS} tasks)...")
    rng = np.random.RandomState(42)
    train_data = generate_dataset(n_train_per_task, train_range, rng)
    print(f"  {len(train_data[0]):,} examples")

    # Verify task balance
    task_counts = {}
    for tid in train_data[0].numpy():
        task_counts[tid] = task_counts.get(tid, 0) + 1
    for task_name in TASK_LIST:
        tid = TASK_ID[task_name]
        print(f"    {task_name:12s}: {task_counts.get(tid, 0):,} examples")

    # Label distribution for 30-class tasks
    for task_name in ['mod30', 'prodmod30', 'summod30']:
        tid = TASK_ID[task_name]
        mask = (train_data[0] == tid)
        task_labels = train_data[2][mask]
        unique, counts = torch.unique(task_labels, return_counts=True)
        print(f"    {task_name:12s} label range: {unique.min().item()}-{unique.max().item()}, "
              f"n_unique={len(unique)}")

    # Generate test data (fixed seeds per range)
    print(f"\nGenerating test data...")
    test_datasets = OrderedDict()
    for range_name, r in test_ranges.items():
        test_rng = np.random.RandomState(hash(range_name) % 2**31)
        data = generate_dataset(n_eval_per_task, r, test_rng)
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

    return all_results, test_datasets


# ============================================================
# PART B: SCALING TEST (d_model=256)
# ============================================================

def run_part_b():
    print("\n" + "=" * 70)
    print("PHASE 11 PART B: SCALING TEST (d_model=256)")
    print("2 configs at larger model scale")
    print("=" * 70)

    d_model = 256
    nhead = 8
    num_layers = 6
    epochs = 30
    batch_size = 512
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

    print(f"\nGenerating training data for scaling test...")
    rng = np.random.RandomState(42)
    train_data = generate_dataset(n_train_per_task, train_range, rng)
    print(f"  {len(train_data[0]):,} examples")

    print(f"Generating test data...")
    test_datasets = OrderedDict()
    for range_name, r in test_ranges.items():
        test_rng = np.random.RandomState(hash(range_name) % 2**31)
        data = generate_dataset(n_eval_per_task, r, test_rng)
        test_datasets[range_name] = data

    scale_configs = OrderedDict([
        ('spirnor_256', EMBED_CONFIGS['spirnor_rational5']),
        ('learned_256', EMBED_CONFIGS['learned']),
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

def print_summary(a_results, b_results):
    range_keys = ['in_range', 'ood_2k_5k', 'ood_5k_20k',
                  'ood_20k_100k', 'ood_100k_500k']
    range_labels = ['In-Range', '2K-5K', '5K-20K', '20K-100K', '100K-500K']

    print("\n" + "=" * 70)
    print("PHASE 11 RESULTS SUMMARY")
    print("=" * 70)

    # ---- Part A: Overall accuracy table ----
    print("\n  PART A: Overall Accuracy (d_model=128)")
    header = f"  {'Config':20s} | {'Params':>10s}"
    for label in range_labels:
        header += f" | {label:>9s}"
    header += f" | {'Avg OOD':>9s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for cfg_name, res in a_results.items():
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

    # ---- Per-Family Analysis at 2K-5K ----
    families = {
        'A': {'name': 'CRT Validation', 'tasks': ['mod30', 'prodmod30']},
        'B': {'name': 'Compositional', 'tasks': ['summod30', 'diophantine', 'gcdequals']},
        'C': {'name': 'Set Reasoning', 'tasks': ['setalldiv', 'setcoprime']},
        'D': {'name': 'Pattern Detection', 'tasks': ['hiddenmod']},
    }

    print("\n" + "-" * 70)
    print("PER-FAMILY ANALYSIS (2K-5K OOD)")
    print("-" * 70)

    for fam_id, fam in families.items():
        print(f"\n  Family {fam_id}: {fam['name']}")
        for cfg_name, res in a_results.items():
            vals = []
            task_strs = []
            for t in fam['tasks']:
                v = res.get('ood_2k_5k', {}).get(t, 0)
                vals.append(v)
                task_strs.append(f"{TASKS[t]['name']}={v:.3f}")
            avg = sum(vals) / len(vals) if vals else 0
            print(f"    {cfg_name:20s}: avg={avg:.4f}  [{', '.join(task_strs)}]")

    # ---- Full Per-Task Breakdown at 2K-5K ----
    print("\n" + "-" * 70)
    print("FULL PER-TASK BREAKDOWN AT 2K-5K")
    print("-" * 70)

    task_header = f"  {'Config':20s}"
    for task_name in TASK_LIST:
        task_header += f" | {TASKS[task_name]['name']:>10s}"
    print(task_header)
    print("  " + "-" * (len(task_header) - 2))

    for cfg_name, res in a_results.items():
        line = f"  {cfg_name:20s}"
        for task_name in TASK_LIST:
            val = res.get('ood_2k_5k', {}).get(task_name, 0)
            line += f" | {val:.4f}    "
        print(line)

    # ---- SPIRNOR vs Learned advantage ----
    if 'spirnor_rational5' in a_results and 'learned' in a_results:
        print("\n" + "-" * 70)
        print("SPIRNOR ADVANTAGE AT 2K-5K OOD")
        print("-" * 70)

        for task_name in TASK_LIST:
            s_val = a_results['spirnor_rational5'].get('ood_2k_5k', {}).get(task_name, 0)
            l_val = a_results['learned'].get('ood_2k_5k', {}).get(task_name, 0)
            if l_val > 0:
                ratio = s_val / l_val
            else:
                ratio = float('inf') if s_val > 0 else 1.0
            diff = s_val - l_val
            print(f"  {TASKS[task_name]['name']:12s}: SPIRNOR={s_val:.4f}, "
                  f"Learned={l_val:.4f}, diff={diff:+.4f}, "
                  f"ratio={ratio:.1f}x")

    # ---- Part B: Scaling Results ----
    if b_results:
        print("\n" + "=" * 70)
        print("PART B: SCALING TEST (d_model=256)")
        print("=" * 70)

        header = f"  {'Config':20s} | {'Params':>10s}"
        for label in range_labels:
            header += f" | {label:>9s}"
        header += f" | {'Avg OOD':>9s}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for cfg_name, res in b_results.items():
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
            ('spirnor_rational5', 'spirnor_256'),
            ('learned', 'learned_256'),
        ]
        for base, scaled in scale_pairs:
            if base not in a_results or scaled not in b_results:
                continue
            b_ood = [a_results[base].get(k, {}).get('overall', 0) for k in range_keys[1:]]
            s_ood = [b_results[scaled].get(k, {}).get('overall', 0) for k in range_keys[1:]]
            b_avg = sum(b_ood) / len(b_ood)
            s_avg = sum(s_ood) / len(s_ood)
            diff = s_avg - b_avg
            print(f"    {base:20s}: {b_avg:.4f} -> {s_avg:.4f} ({diff:+.4f})")

    # ---- Key Findings ----
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Best config overall
    for cfg_name, res in a_results.items():
        in_acc = res.get('in_range', {}).get('overall', 0)
        ood_vals = [res.get(k, {}).get('overall', 0) for k in range_keys[1:]]
        avg_ood = sum(ood_vals) / len(ood_vals) if ood_vals else 0
        print(f"  {cfg_name:20s}: in_range={in_acc:.4f}, avg_OOD={avg_ood:.4f}")

    # CRT validation
    if 'spirnor_rational5' in a_results:
        mod30_ood = a_results['spirnor_rational5'].get('ood_100k_500k', {}).get('mod30', 0)
        print(f"\n  CRT Validation: Mod30 @ 100K-500K = {mod30_ood:.4f} "
              f"({'PASS' if mod30_ood > 0.90 else 'INVESTIGATE'})")

    # Compositionality gradient
    if 'spirnor_rational5' in a_results:
        tasks_gradient = ['mod30', 'prodmod30', 'summod30', 'diophantine', 'gcdequals']
        print(f"\n  Compositionality Gradient (SPIRNOR @ 2K-5K):")
        for t in tasks_gradient:
            v = a_results['spirnor_rational5'].get('ood_2k_5k', {}).get(t, 0)
            bar = '#' * int(v * 30)
            print(f"    {TASKS[t]['name']:12s}: {v:.4f} {bar}")


def save_results(a_results, b_results):
    def clean(obj):
        if isinstance(obj, (OrderedDict, dict)):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    save_data = {
        'phase': 'Phase 11: Compositional & Structured Numeric Reasoning',
        'config': {
            'part_a': {
                'd_model': 128, 'nhead': 4, 'num_layers': 6, 'epochs': 30,
                'train_range': [2, 2000], 'n_train_per_task': 50000,
                'n_eval_per_task': 1000, 'n_tasks': N_TASKS,
                'tasks': list(TASK_LIST),
                'spirnor_constants': RATIONAL_5_NAMES,
            },
            'part_b': {
                'd_model': 256, 'nhead': 8, 'num_layers': 6, 'epochs': 30,
                'train_range': [2, 2000], 'n_train_per_task': 50000,
                'n_eval_per_task': 1000,
            },
        },
        'task_families': {
            'A_CRT': ['mod30', 'prodmod30'],
            'B_Compositional': ['summod30', 'diophantine', 'gcdequals'],
            'C_Set': ['setalldiv', 'setcoprime'],
            'D_Pattern': ['hiddenmod'],
        },
        'results_part_a': clean(a_results),
        'results_part_b': clean(b_results) if b_results else {},
    }

    save_path = 'SPIRNOR_AI_PHASE11_RESULTS.json'
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {save_path}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    total_start = time.time()

    # Part A: 2 configs x 8 tasks at d_model=128
    a_results, test_datasets = run_part_a()

    # Part B: 2 configs at d_model=256
    b_results = run_part_b()

    # Summary and save
    print_summary(a_results, b_results)
    save_results(a_results, b_results)

    total_time = time.time() - total_start
    print(f"\nTotal experiment time: {total_time / 60:.1f} minutes")
    print("Phase 11 complete!")
