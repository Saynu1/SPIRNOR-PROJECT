#!/usr/bin/env python3
"""
SPIRNOR AI Phase 13: Baseline Comparisons — SPIRNOR vs. Established Encoding Methods

6-way comparison on Phase 12 temporal tasks:
  1. spirnor_temporal  — SPIRNOR with period-matched constants (our method)
  2. learned           — nn.Embedding baseline
  3. rff               — Random Fourier Features (Tancik et al. 2020)
  4. sinusoidal        — Vaswani-style geometric frequency encoding
  5. time2vec          — Time2Vec learned frequencies (Kazemi et al. 2019)
  6. learned_fourier   — Learned sin/cos frequencies

Core question: Does SPIRNOR's advantage come from having MATCHED frequencies
(2pi/T for known periods T), or just from having any sinusoidal encoding?

Same 8 temporal tasks, same architecture, same training procedure as Phase 12.
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

# Temporal constants: 2pi/T for temporal periods T (in hours)
TEMPORAL_PERIODS = [24, 168, 720, 2190, 8766]  # daily, weekly, monthly, quarterly, yearly
TEMPORAL_VALUES = [TWO_PI / T for T in TEMPORAL_PERIODS]
TEMPORAL_NAMES = [f'2pi/{T}' for T in TEMPORAL_PERIODS]

# Temporal reference: t=1 is Monday Jan 1 01:00 (t=0 reserved for padding)
HOURS_PER_DAY = 24
HOURS_PER_WEEK = 168  # 7 * 24
DAYS_PER_YEAR = 365   # approximate

# Season boundaries (day of year, 0-indexed)
SEASON_SPRING_START = 80
SEASON_SUMMER_START = 172
SEASON_FALL_START = 266
SEASON_WINTER2_START = 355

MAX_SEQ = 8

# ============================================================
# TASK DEFINITIONS (8 tasks across 3 families — same as Phase 12)
# ============================================================

TASKS = OrderedDict([
    ('hourofday',      {'n_inputs': 1, 'n_classes': 24, 'name': 'HourOfDay',      'family': 'A'}),
    ('dayofweek',      {'n_inputs': 1, 'n_classes': 7,  'name': 'DayOfWeek',      'family': 'A'}),
    ('isweekend',      {'n_inputs': 1, 'n_classes': 2,  'name': 'IsWeekend',      'family': 'B'}),
    ('timeslot',       {'n_inputs': 1, 'n_classes': 4,  'name': 'TimeSlot',       'family': 'B'}),
    ('isbusinesshour', {'n_inputs': 1, 'n_classes': 2,  'name': 'IsBusinessHr',   'family': 'B'}),
    ('season',         {'n_inputs': 1, 'n_classes': 4,  'name': 'Season',         'family': 'B'}),
    ('samedow',        {'n_inputs': 2, 'n_classes': 2,  'name': 'SameDOW',        'family': 'C'}),
    ('hourdiffmod12',  {'n_inputs': 2, 'n_classes': 12, 'name': 'HrDiffMod12',    'family': 'C'}),
])

TASK_LIST = list(TASKS.keys())
TASK_ID = {name: i for i, name in enumerate(TASK_LIST)}
N_TASKS = len(TASK_LIST)


# ============================================================
# TEMPORAL HELPERS
# ============================================================

def hour_of_day(t):
    return t % HOURS_PER_DAY

def day_of_week(t):
    return (t // HOURS_PER_DAY) % 7

def day_of_year(t):
    return (t // HOURS_PER_DAY) % DAYS_PER_YEAR

def get_season(t):
    doy = day_of_year(t)
    if doy < SEASON_SPRING_START or doy >= SEASON_WINTER2_START:
        return 0  # Winter
    elif doy < SEASON_SUMMER_START:
        return 1  # Spring
    elif doy < SEASON_FALL_START:
        return 2  # Summer
    else:
        return 3  # Fall

def is_weekend(t):
    return 1 if day_of_week(t) >= 5 else 0

def is_business_hour(t):
    dow = day_of_week(t)
    h = hour_of_day(t)
    return 1 if (dow < 5 and 9 <= h < 17) else 0

def time_slot(t):
    h = hour_of_day(t)
    return h // 6


# ============================================================
# EMBEDDINGS (6 types)
# ============================================================

# --- 1. SPIRNOR Embedding (our method) ---

class SPIRNORNumericEmbedding(nn.Module):
    """SPIRNOR embedding with temporal constants.
    7 features per constant: [r*sin(θ)*cos(φ), r*sin(θ)*sin(φ), r*cos(θ),
                              sin(θ), cos(θ), sin(φ), cos(φ)]
    where θ = C*n mod 2π, φ = PHI*n mod 2π, r = ln(n).
    """
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


# --- 2. Learned Embedding (standard baseline) ---

class LearnedNumericEmbedding(nn.Module):
    def __init__(self, max_num, d_model):
        super().__init__()
        self.max_num = max_num
        self.embed = nn.Embedding(max_num + 1, d_model)

    def forward(self, numbers):
        idx = numbers.long() % (self.max_num + 1)
        return self.embed(idx)


# --- 3. Random Fourier Features (Tancik et al. 2020) ---

class RandomFourierEmbedding(nn.Module):
    """Random Fourier Features: sin/cos with random Gaussian frequencies.
    features = [sin(2π·σ_i·x), cos(2π·σ_i·x)] where σ_i ~ N(0, σ²).
    Frequencies are fixed at initialization (not learned).
    """
    def __init__(self, d_model, n_freq=18, sigma=0.01, seed=123):
        super().__init__()
        rng = torch.Generator()
        rng.manual_seed(seed)
        freqs = torch.randn(n_freq, generator=rng) * sigma
        self.register_buffer('freqs', freqs)
        raw_dim = n_freq * 2  # sin + cos pairs
        self.proj = nn.Linear(raw_dim, d_model)
        self.pad_embed = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(self, numbers):
        mask = (numbers > 0)
        x = numbers.float().clamp(min=1.0)
        # x: [B, seq], freqs: [n_freq] -> angles: [B, seq, n_freq]
        angles = TWO_PI * x.unsqueeze(-1) * self.freqs
        features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        embedded = self.proj(features)

        pad = self.pad_embed.view(1, 1, -1).expand_as(embedded)
        embedded = torch.where(mask.unsqueeze(-1), embedded, pad)
        return embedded


# --- 4. Sinusoidal Value Embedding (Vaswani-style, adapted for values) ---

class SinusoidalValueEmbedding(nn.Module):
    """Vaswani-style geometric frequency encoding adapted for numeric values.
    features_{2i} = sin(x / base^(2i/d_raw))
    features_{2i+1} = cos(x / base^(2i/d_raw))
    Geometric frequency spacing from fast (short period) to slow (long period).
    """
    def __init__(self, d_model, n_freq=18, base=10000.0):
        super().__init__()
        # Geometric frequencies: 1/base^(2i/d_raw) where d_raw = 2*n_freq
        d_raw = 2 * n_freq
        inv_freq = 1.0 / (base ** (torch.arange(0, d_raw, 2).float() / d_raw))
        self.register_buffer('inv_freq', inv_freq)
        raw_dim = n_freq * 2
        self.proj = nn.Linear(raw_dim, d_model)
        self.pad_embed = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(self, numbers):
        mask = (numbers > 0)
        x = numbers.float().clamp(min=1.0)
        # x: [B, seq], inv_freq: [n_freq] -> angles: [B, seq, n_freq]
        angles = x.unsqueeze(-1) * self.inv_freq
        features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        embedded = self.proj(features)

        pad = self.pad_embed.view(1, 1, -1).expand_as(embedded)
        embedded = torch.where(mask.unsqueeze(-1), embedded, pad)
        return embedded


# --- 5. Time2Vec (Kazemi et al. 2019) ---

class Time2VecEmbedding(nn.Module):
    """Time2Vec: 1 linear + k sinusoidal features with learned frequencies.
    t2v(x)[0] = w_0 * x + phi_0          (linear trend)
    t2v(x)[i] = sin(w_i * x + phi_i)     for i > 0  (periodic)
    All w_i and phi_i are learned during training.
    """
    def __init__(self, d_model, n_features=36):
        super().__init__()
        self.n_features = n_features
        # Learned frequencies and phases
        self.w = nn.Parameter(torch.empty(n_features))
        self.phi = nn.Parameter(torch.empty(n_features))
        # Initialize: small frequencies centered on temporal range
        nn.init.uniform_(self.w, -0.01, 0.01)
        nn.init.uniform_(self.phi, 0, TWO_PI)
        self.proj = nn.Linear(n_features, d_model)
        self.pad_embed = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(self, numbers):
        mask = (numbers > 0)
        x = numbers.float().clamp(min=1.0)
        # x: [B, seq] -> [B, seq, n_features]
        raw = x.unsqueeze(-1) * self.w + self.phi
        # First feature is linear, rest are sin
        features = torch.cat([
            raw[..., :1],                    # linear component
            torch.sin(raw[..., 1:]),         # sinusoidal components
        ], dim=-1)
        embedded = self.proj(features)

        pad = self.pad_embed.view(1, 1, -1).expand_as(embedded)
        embedded = torch.where(mask.unsqueeze(-1), embedded, pad)
        return embedded


# --- 6. Learned Fourier Features ---

class LearnedFourierEmbedding(nn.Module):
    """Learned Fourier Features: sin/cos with learned frequencies and phases.
    features = [sin(w_i * x + phi_i), cos(w_i * x + phi_i)]
    All w_i and phi_i are learned during training.
    """
    def __init__(self, d_model, n_freq=18):
        super().__init__()
        self.n_freq = n_freq
        self.w = nn.Parameter(torch.empty(n_freq))
        self.phi = nn.Parameter(torch.empty(n_freq))
        # Initialize: small random frequencies, random phases
        nn.init.normal_(self.w, 0.0, 0.01)
        nn.init.uniform_(self.phi, 0, TWO_PI)
        raw_dim = n_freq * 2
        self.proj = nn.Linear(raw_dim, d_model)
        self.pad_embed = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(self, numbers):
        mask = (numbers > 0)
        x = numbers.float().clamp(min=1.0)
        # x: [B, seq], w/phi: [n_freq] -> angles: [B, seq, n_freq]
        angles = x.unsqueeze(-1) * self.w + self.phi
        features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        embedded = self.proj(features)

        pad = self.pad_embed.view(1, 1, -1).expand_as(embedded)
        embedded = torch.where(mask.unsqueeze(-1), embedded, pad)
        return embedded


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
                 task_configs, max_num=17520, max_len=10,
                 dropout=0.1, const_values=None, const_names=None):
        super().__init__()
        self.d_model = d_model
        self.embed_type = embed_type

        if embed_type == 'learned':
            self.num_embed = LearnedNumericEmbedding(max_num, d_model)
        elif embed_type == 'spirnor':
            self.num_embed = SPIRNORNumericEmbedding(
                d_model, const_values, const_names)
        elif embed_type == 'rff':
            self.num_embed = RandomFourierEmbedding(d_model, n_freq=18, sigma=0.01)
        elif embed_type == 'sinusoidal':
            self.num_embed = SinusoidalValueEmbedding(d_model, n_freq=18, base=10000.0)
        elif embed_type == 'time2vec':
            self.num_embed = Time2VecEmbedding(d_model, n_features=36)
        elif embed_type == 'learned_fourier':
            self.num_embed = LearnedFourierEmbedding(d_model, n_freq=18)
        else:
            raise ValueError(f"Unknown embed_type: {embed_type}")

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
# DATA GENERATION (identical to Phase 12)
# ============================================================

def generate_dataset(n_per_task, time_range, rng):
    lo, hi = time_range
    all_task_ids = []
    all_numbers = []
    all_labels = []

    for task_name in TASK_LIST:
        task_id = TASK_ID[task_name]

        if task_name == 'hourofday':
            for _ in range(n_per_task):
                t = int(rng.randint(lo, hi))
                nums = [t] + [0] * (MAX_SEQ - 1)
                label = hour_of_day(t)
                all_task_ids.append(task_id)
                all_numbers.append(nums)
                all_labels.append(label)

        elif task_name == 'dayofweek':
            for _ in range(n_per_task):
                t = int(rng.randint(lo, hi))
                nums = [t] + [0] * (MAX_SEQ - 1)
                label = day_of_week(t)
                all_task_ids.append(task_id)
                all_numbers.append(nums)
                all_labels.append(label)

        elif task_name == 'isweekend':
            n_each = n_per_task // 2
            count = 0
            attempts = 0
            while count < n_each and attempts < n_per_task * 20:
                attempts += 1
                t = int(rng.randint(lo, hi))
                if day_of_week(t) >= 5:
                    nums = [t] + [0] * (MAX_SEQ - 1)
                    all_task_ids.append(task_id)
                    all_numbers.append(nums)
                    all_labels.append(1)
                    count += 1
            count = 0
            attempts = 0
            while count < n_per_task - n_each and attempts < n_per_task * 20:
                attempts += 1
                t = int(rng.randint(lo, hi))
                if day_of_week(t) < 5:
                    nums = [t] + [0] * (MAX_SEQ - 1)
                    all_task_ids.append(task_id)
                    all_numbers.append(nums)
                    all_labels.append(0)
                    count += 1

        elif task_name == 'timeslot':
            for _ in range(n_per_task):
                t = int(rng.randint(lo, hi))
                nums = [t] + [0] * (MAX_SEQ - 1)
                label = time_slot(t)
                all_task_ids.append(task_id)
                all_numbers.append(nums)
                all_labels.append(label)

        elif task_name == 'isbusinesshour':
            n_each = n_per_task // 2
            count = 0
            attempts = 0
            while count < n_each and attempts < n_per_task * 20:
                attempts += 1
                t = int(rng.randint(lo, hi))
                if is_business_hour(t):
                    nums = [t] + [0] * (MAX_SEQ - 1)
                    all_task_ids.append(task_id)
                    all_numbers.append(nums)
                    all_labels.append(1)
                    count += 1
            count = 0
            attempts = 0
            while count < n_per_task - n_each and attempts < n_per_task * 20:
                attempts += 1
                t = int(rng.randint(lo, hi))
                if not is_business_hour(t):
                    nums = [t] + [0] * (MAX_SEQ - 1)
                    all_task_ids.append(task_id)
                    all_numbers.append(nums)
                    all_labels.append(0)
                    count += 1

        elif task_name == 'season':
            for _ in range(n_per_task):
                t = int(rng.randint(lo, hi))
                nums = [t] + [0] * (MAX_SEQ - 1)
                label = get_season(t)
                all_task_ids.append(task_id)
                all_numbers.append(nums)
                all_labels.append(label)

        elif task_name == 'samedow':
            n_each = n_per_task // 2
            count = 0
            attempts = 0
            while count < n_each and attempts < n_per_task * 20:
                attempts += 1
                t1 = int(rng.randint(lo, hi))
                dow1 = day_of_week(t1)
                t2 = int(rng.randint(lo, hi))
                if day_of_week(t2) == dow1 and t2 != t1:
                    nums = [t1, t2] + [0] * (MAX_SEQ - 2)
                    all_task_ids.append(task_id)
                    all_numbers.append(nums)
                    all_labels.append(1)
                    count += 1
            count = 0
            attempts = 0
            while count < n_per_task - n_each and attempts < n_per_task * 20:
                attempts += 1
                t1 = int(rng.randint(lo, hi))
                t2 = int(rng.randint(lo, hi))
                if day_of_week(t1) != day_of_week(t2):
                    nums = [t1, t2] + [0] * (MAX_SEQ - 2)
                    all_task_ids.append(task_id)
                    all_numbers.append(nums)
                    all_labels.append(0)
                    count += 1

        elif task_name == 'hourdiffmod12':
            for _ in range(n_per_task):
                t1 = int(rng.randint(lo, hi))
                t2 = int(rng.randint(lo, hi))
                h1 = hour_of_day(t1)
                h2 = hour_of_day(t2)
                label = (h2 - h1) % 12
                nums = [t1, t2] + [0] * (MAX_SEQ - 2)
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
# EMBEDDING CONFIGS (6 total)
# ============================================================

EMBED_CONFIGS = OrderedDict([
    ('spirnor_temporal', {
        'type': 'spirnor',
        'desc': 'SPIRNOR: period-matched {2pi/24,168,720,2190,8766}',
        'const_values': TEMPORAL_VALUES,
        'const_names': TEMPORAL_NAMES,
    }),
    ('learned', {
        'type': 'learned',
        'desc': 'Learned: nn.Embedding(17521, d_model)',
        'const_values': None,
        'const_names': None,
    }),
    ('rff', {
        'type': 'rff',
        'desc': 'RFF: 18 random Gaussian freqs (sigma=0.01)',
        'const_values': None,
        'const_names': None,
    }),
    ('sinusoidal', {
        'type': 'sinusoidal',
        'desc': 'Sinusoidal: Vaswani-style geometric freqs (base=10000)',
        'const_values': None,
        'const_names': None,
    }),
    ('time2vec', {
        'type': 'time2vec',
        'desc': 'Time2Vec: 1 linear + 35 learned sin (Kazemi 2019)',
        'const_values': None,
        'const_names': None,
    }),
    ('learned_fourier', {
        'type': 'learned_fourier',
        'desc': 'Learned Fourier: 18 learned sin/cos pairs',
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
        max_len=MAX_SEQ + 2,
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
# MAIN EXPERIMENT (d_model=128, 6 configs, 8 tasks)
# ============================================================

def run_experiment():
    print("=" * 70)
    print("PHASE 13: BASELINE COMPARISONS (d_model=128)")
    print("6 embedding configs x 8 temporal tasks")
    print("=" * 70)

    d_model = 128
    nhead = 4
    num_layers = 6
    epochs = 30
    batch_size = 1024
    lr = 3e-4

    train_range = (1, 17521)
    n_train_per_task = 50000
    n_eval_per_task = 1000
    max_train = train_range[1] - 1  # 17520

    test_ranges = OrderedDict([
        ('in_range',      (1, 17521)),
        ('ood_2y_5y',     (17521, 43801)),
        ('ood_5y_10y',    (43801, 87601)),
        ('ood_10y_50y',   (87601, 438001)),
    ])

    # Generate training data
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
        print(f"    {task_name:15s}: {task_counts.get(tid, 0):,} examples")

    # Temporal sanity check
    print("\n  Temporal sanity check:")
    test_stamps = [1, 24, 120, 168, 17520]
    for t in test_stamps:
        print(f"    t={t:6d}: hour={hour_of_day(t):2d}, "
              f"dow={day_of_week(t)} ({'Wknd' if is_weekend(t) else 'Wkdy'}), "
              f"slot={time_slot(t)}, season={get_season(t)}, "
              f"biz={is_business_hour(t)}")

    # Generate test data
    print(f"\nGenerating test data...")
    test_datasets = OrderedDict()
    for range_name, r in test_ranges.items():
        test_rng = np.random.RandomState(hash(range_name) % 2**31)
        data = generate_dataset(n_eval_per_task, r, test_rng)
        test_datasets[range_name] = data
        print(f"  {range_name:18s}: {len(data[0]):,} examples "
              f"(hours {r[0]:,}-{r[1]:,})")

    # Run all 6 configurations
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
            max_num=max_train
        )
        all_results[cfg_name] = result

    return all_results


# ============================================================
# RESULTS SUMMARY
# ============================================================

def print_summary(results):
    range_keys = ['in_range', 'ood_2y_5y', 'ood_5y_10y', 'ood_10y_50y']
    range_labels = ['In-Range', '2-5 Yrs', '5-10 Yrs', '10-50 Yrs']

    print("\n" + "=" * 70)
    print("PHASE 13 RESULTS SUMMARY: 6-WAY BASELINE COMPARISON")
    print("=" * 70)

    # ---- Overall accuracy table ----
    print("\n  OVERALL ACCURACY (sorted by avg OOD)")
    header = f"  {'Config':20s} | {'Params':>10s}"
    for label in range_labels:
        header += f" | {label:>9s}"
    header += f" | {'Avg OOD':>9s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Compute avg OOD for sorting
    config_ood = {}
    for cfg_name, res in results.items():
        ood_vals = [res.get(k, {}).get('overall', 0) for k in range_keys[1:]]
        config_ood[cfg_name] = sum(ood_vals) / len(ood_vals) if ood_vals else 0

    sorted_configs = sorted(results.keys(), key=lambda c: config_ood[c], reverse=True)

    for cfg_name in sorted_configs:
        res = results[cfg_name]
        line = f"  {cfg_name:20s} | {res['params']:10,}"
        for key in range_keys:
            val = res.get(key, {}).get('overall', 0)
            line += f" | {val:.4f}   "
        line += f" | {config_ood[cfg_name]:.4f}   "
        print(line)

    # ---- Per-task at 2-5yr OOD ----
    print("\n" + "-" * 70)
    print("PER-TASK ACCURACY AT 2-5 YEARS OOD (sorted by avg OOD)")
    print("-" * 70)

    task_header = f"  {'Config':20s}"
    for task_name in TASK_LIST:
        task_header += f" | {TASKS[task_name]['name']:>10s}"
    task_header += f" | {'Overall':>8s}"
    print(task_header)
    print("  " + "-" * (len(task_header) - 2))

    for cfg_name in sorted_configs:
        res = results[cfg_name]
        line = f"  {cfg_name:20s}"
        for task_name in TASK_LIST:
            val = res.get('ood_2y_5y', {}).get(task_name, 0)
            line += f" | {val:.4f}    "
        ov = res.get('ood_2y_5y', {}).get('overall', 0)
        line += f" | {ov:.4f}  "
        print(line)

    # ---- SPIRNOR advantage over each baseline ----
    if 'spirnor_temporal' in results:
        print("\n" + "-" * 70)
        print("SPIRNOR ADVANTAGE (avg OOD) OVER EACH BASELINE")
        print("-" * 70)

        spirnor_ood = config_ood['spirnor_temporal']
        for cfg_name in sorted_configs:
            if cfg_name == 'spirnor_temporal':
                continue
            other_ood = config_ood[cfg_name]
            diff = spirnor_ood - other_ood
            if other_ood > 0:
                ratio = spirnor_ood / other_ood
            else:
                ratio = float('inf')
            print(f"  vs {cfg_name:20s}: SPIRNOR {spirnor_ood:.4f} vs {other_ood:.4f}, "
                  f"diff={diff:+.4f}, ratio={ratio:.2f}x")

    # ---- Per-task SPIRNOR vs each baseline at 2-5yr ----
    if 'spirnor_temporal' in results:
        print("\n" + "-" * 70)
        print("PER-TASK SPIRNOR vs BASELINES AT 2-5yr OOD")
        print("-" * 70)

        for task_name in TASK_LIST:
            s_val = results['spirnor_temporal'].get('ood_2y_5y', {}).get(task_name, 0)
            best_baseline = None
            best_val = -1
            for cfg_name in sorted_configs:
                if cfg_name == 'spirnor_temporal':
                    continue
                v = results[cfg_name].get('ood_2y_5y', {}).get(task_name, 0)
                if v > best_val:
                    best_val = v
                    best_baseline = cfg_name
            diff = s_val - best_val
            print(f"  {TASKS[task_name]['name']:12s}: SPIRNOR={s_val:.4f}, "
                  f"best_baseline={best_val:.4f} ({best_baseline}), "
                  f"gap={diff:+.4f}")

    # ---- Encoding category analysis ----
    print("\n" + "-" * 70)
    print("ENCODING CATEGORY ANALYSIS (avg OOD)")
    print("-" * 70)

    categories = OrderedDict([
        ('Period-Matched Fixed', ['spirnor_temporal']),
        ('Geometric Fixed', ['sinusoidal']),
        ('Random Fixed', ['rff']),
        ('Learned Frequencies', ['time2vec', 'learned_fourier']),
        ('Learned Lookup', ['learned']),
    ])

    for cat_name, configs in categories.items():
        present = [c for c in configs if c in results]
        if not present:
            continue
        avg = sum(config_ood[c] for c in present) / len(present)
        cfg_strs = [f"{c}={config_ood[c]:.4f}" for c in present]
        print(f"  {cat_name:25s}: avg={avg:.4f}  [{', '.join(cfg_strs)}]")

    # ---- Key findings ----
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    print(f"\n  1. Ranking (avg OOD):")
    for i, cfg_name in enumerate(sorted_configs):
        print(f"     #{i+1}: {cfg_name:20s} = {config_ood[cfg_name]:.4f}")

    spirnor_rank = sorted_configs.index('spirnor_temporal') + 1
    print(f"\n  2. SPIRNOR rank: #{spirnor_rank} of {len(sorted_configs)}")

    sinusoidal_methods = ['spirnor_temporal', 'rff', 'sinusoidal', 'time2vec', 'learned_fourier']
    sin_present = [c for c in sinusoidal_methods if c in config_ood]
    if sin_present:
        sin_avg = sum(config_ood[c] for c in sin_present) / len(sin_present)
        learned_ood = config_ood.get('learned', 0)
        print(f"\n  3. Sinusoidal methods avg: {sin_avg:.4f} vs Learned: {learned_ood:.4f}")
        if learned_ood > 0:
            print(f"     All sinusoidal >> Learned: {sin_avg/learned_ood:.2f}x")

    print(f"\n  4. Temporal CRT validation (SPIRNOR @ 10-50yr OOD):")
    if 'spirnor_temporal' in results:
        for t in ['hourofday', 'dayofweek']:
            v = results['spirnor_temporal'].get('ood_10y_50y', {}).get(t, 0)
            print(f"     {TASKS[t]['name']:12s}: {v:.4f}")


def save_results(results):
    def clean(obj):
        if isinstance(obj, (OrderedDict, dict)):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    save_data = {
        'phase': 'Phase 13: Baseline Comparisons',
        'config': {
            'd_model': 128, 'nhead': 4, 'num_layers': 6, 'epochs': 30,
            'train_range': [1, 17521], 'n_train_per_task': 50000,
            'n_eval_per_task': 1000, 'n_tasks': N_TASKS,
            'tasks': list(TASK_LIST),
            'spirnor_constants': TEMPORAL_NAMES,
            'temporal_periods': TEMPORAL_PERIODS,
            'n_configs': len(EMBED_CONFIGS),
            'configs': list(EMBED_CONFIGS.keys()),
        },
        'baselines': {
            'spirnor_temporal': 'Period-matched fixed: {2pi/24, 2pi/168, 2pi/720, 2pi/2190, 2pi/8766}',
            'learned': 'nn.Embedding(17521, d_model) with mod-wrapping OOD',
            'rff': 'Random Fourier Features: 18 freqs ~ N(0, 0.01^2), Tancik 2020',
            'sinusoidal': 'Vaswani-style geometric freqs: 1/10000^(2i/d), 18 pairs',
            'time2vec': 'Time2Vec: 1 linear + 35 learned sin, Kazemi 2019',
            'learned_fourier': 'Learned Fourier: 18 learned sin/cos pairs',
        },
        'results': clean(results),
    }

    save_path = 'SPIRNOR_AI_PHASE13_RESULTS.json'
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {save_path}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    total_start = time.time()

    results = run_experiment()

    print_summary(results)
    save_results(results)

    total_time = time.time() - total_start
    print(f"\nTotal experiment time: {total_time / 60:.1f} minutes")
    print("Phase 13 complete!")
