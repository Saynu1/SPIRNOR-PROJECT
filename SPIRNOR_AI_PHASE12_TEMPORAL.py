#!/usr/bin/env python3
"""
SPIRNOR AI Phase 12: Temporal/Periodic Data

First domain-specific SPIRNOR application: temporal pattern recognition.
Core insight: temporal data has natural periodic structure (hours, days, weeks,
seasons). SPIRNOR constants matched to temporal periods (2pi/24, 2pi/168, etc.)
create exact periodic encoding of timestamps.

8 Tasks across 3 Families:
  A. Direct Temporal Readout:     HourOfDay(t), DayOfWeek(t)
  B. Hierarchical Composition:    IsWeekend(t), TimeSlot(t), IsBusinessHour(t), SeasonOfYear(t)
  C. Multi-Timestamp Reasoning:   SameDayOfWeek(t1,t2), HourDiffMod12(t1,t2)

2 embedding configs: SPIRNOR temporal vs Learned baseline
Training: 400K examples (50K per task x 8 tasks), train t in [0, 17520) (2 years of hours)
Eval ranges: in_range, 2-5 years, 5-10 years, 10-50 years

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
# All timestamps offset by 1 to avoid t=0 collision with padding value
HOURS_PER_DAY = 24
HOURS_PER_WEEK = 168  # 7 * 24
DAYS_PER_YEAR = 365  # approximate (ignoring leap years for simplicity)

# Season boundaries (day of year, 0-indexed)
# Winter: Dec 21 (day 355) - Mar 19 (day 79)
# Spring: Mar 20 (day 80) - Jun 19 (day 171)
# Summer: Jun 20 (day 172) - Sep 21 (day 265)
# Fall:   Sep 22 (day 266) - Dec 20 (day 354)
SEASON_SPRING_START = 80
SEASON_SUMMER_START = 172
SEASON_FALL_START = 266
SEASON_WINTER2_START = 355  # Winter wraps around year boundary

MAX_SEQ = 8  # Maximum numbers per example (same format as Phase 11)

# ============================================================
# TASK DEFINITIONS (8 tasks across 3 families)
# ============================================================

TASKS = OrderedDict([
    # Family A: Direct Temporal Readout
    ('hourofday',      {'n_inputs': 1, 'n_classes': 24, 'name': 'HourOfDay',      'family': 'A'}),
    ('dayofweek',      {'n_inputs': 1, 'n_classes': 7,  'name': 'DayOfWeek',      'family': 'A'}),
    # Family B: Hierarchical Temporal Composition
    ('isweekend',      {'n_inputs': 1, 'n_classes': 2,  'name': 'IsWeekend',      'family': 'B'}),
    ('timeslot',       {'n_inputs': 1, 'n_classes': 4,  'name': 'TimeSlot',       'family': 'B'}),
    ('isbusinesshour', {'n_inputs': 1, 'n_classes': 2,  'name': 'IsBusinessHr',   'family': 'B'}),
    ('season',         {'n_inputs': 1, 'n_classes': 4,  'name': 'Season',         'family': 'B'}),
    # Family C: Multi-Timestamp Reasoning
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
    """Extract hour of day (0-23) from timestamp."""
    return t % HOURS_PER_DAY


def day_of_week(t):
    """Extract day of week (0=Mon, 6=Sun) from timestamp."""
    return (t // HOURS_PER_DAY) % 7


def day_of_year(t):
    """Extract approximate day of year (0-364) from timestamp."""
    return (t // HOURS_PER_DAY) % DAYS_PER_YEAR


def get_season(t):
    """Get season (0=winter, 1=spring, 2=summer, 3=fall) from timestamp."""
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
    """Check if timestamp falls on weekend (Sat=5 or Sun=6)."""
    return 1 if day_of_week(t) >= 5 else 0


def is_business_hour(t):
    """Check if timestamp is a business hour (Mon-Fri, 9:00-16:59)."""
    dow = day_of_week(t)
    h = hour_of_day(t)
    return 1 if (dow < 5 and 9 <= h < 17) else 0


def time_slot(t):
    """Map hour to time slot: 0=night(0-5), 1=morning(6-11), 2=afternoon(12-17), 3=evening(18-23)."""
    h = hour_of_day(t)
    return h // 6


# ============================================================
# EMBEDDINGS
# ============================================================

class SPIRNORNumericEmbedding(nn.Module):
    """SPIRNOR embedding with temporal constants."""
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

def generate_dataset(n_per_task, time_range, rng):
    """Generate training/eval data for all 8 Phase 12 temporal tasks.

    Args:
        n_per_task: Number of examples per task
        time_range: (lo, hi) tuple of timestamp range (exclusive hi)
        rng: numpy RandomState
    """
    lo, hi = time_range
    all_task_ids = []
    all_numbers = []
    all_labels = []

    for task_name in TASK_LIST:
        task_id = TASK_ID[task_name]

        if task_name == 'hourofday':
            # t mod 24 -> 24 classes
            for _ in range(n_per_task):
                t = int(rng.randint(lo, hi))
                nums = [t] + [0] * (MAX_SEQ - 1)
                label = hour_of_day(t)
                all_task_ids.append(task_id)
                all_numbers.append(nums)
                all_labels.append(label)

        elif task_name == 'dayofweek':
            # floor(t/24) mod 7 -> 7 classes
            for _ in range(n_per_task):
                t = int(rng.randint(lo, hi))
                nums = [t] + [0] * (MAX_SEQ - 1)
                label = day_of_week(t)
                all_task_ids.append(task_id)
                all_numbers.append(nums)
                all_labels.append(label)

        elif task_name == 'isweekend':
            # Balanced: 50% weekend, 50% weekday
            n_each = n_per_task // 2

            # Positive: weekend timestamps (DoW = 5 or 6)
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

            # Negative: weekday timestamps (DoW = 0-4)
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
            # floor((t mod 24) / 6) -> 4 classes (night/morning/afternoon/evening)
            for _ in range(n_per_task):
                t = int(rng.randint(lo, hi))
                nums = [t] + [0] * (MAX_SEQ - 1)
                label = time_slot(t)
                all_task_ids.append(task_id)
                all_numbers.append(nums)
                all_labels.append(label)

        elif task_name == 'isbusinesshour':
            # Balanced: 50% business hour, 50% non-business hour
            n_each = n_per_task // 2

            # Positive: business hours (Mon-Fri, 9-16)
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

            # Negative: non-business hours
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
            # Season from day-of-year -> 4 classes
            for _ in range(n_per_task):
                t = int(rng.randint(lo, hi))
                nums = [t] + [0] * (MAX_SEQ - 1)
                label = get_season(t)
                all_task_ids.append(task_id)
                all_numbers.append(nums)
                all_labels.append(label)

        elif task_name == 'samedow':
            # Same day of week? Balanced 50/50
            n_each = n_per_task // 2

            # Positive: same day of week
            count = 0
            attempts = 0
            while count < n_each and attempts < n_per_task * 20:
                attempts += 1
                t1 = int(rng.randint(lo, hi))
                dow1 = day_of_week(t1)
                # Generate t2 with same DoW: add k*168 to a base with same DoW
                # Find a random t2 in range with same DoW
                t2 = int(rng.randint(lo, hi))
                if day_of_week(t2) == dow1 and t2 != t1:
                    nums = [t1, t2] + [0] * (MAX_SEQ - 2)
                    all_task_ids.append(task_id)
                    all_numbers.append(nums)
                    all_labels.append(1)
                    count += 1

            # Negative: different day of week
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
            # (hour(t2) - hour(t1)) mod 12 -> 12 classes
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
    ('spirnor_temporal', {
        'type': 'spirnor',
        'desc': 'SPIRNOR with temporal constants {2pi/24,168,720,2190,8766}',
        'const_values': TEMPORAL_VALUES,
        'const_names': TEMPORAL_NAMES,
    }),
    ('learned', {
        'type': 'learned',
        'desc': 'Learned embedding (nn.Embedding baseline)',
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
# PART A: MAIN EXPERIMENT (d_model=128, 2 configs, 8 tasks)
# ============================================================

def run_part_a():
    print("=" * 70)
    print("PHASE 12 PART A: TEMPORAL PATTERN RECOGNITION (d_model=128)")
    print("2 embedding configs x 8 temporal tasks")
    print("=" * 70)

    d_model = 128
    nhead = 4
    num_layers = 6
    epochs = 30
    batch_size = 1024
    lr = 3e-4

    # t starts at 1 (t=0 is padding). Range [1, 17521) = 17520 hours = 2 years
    train_range = (1, 17521)
    n_train_per_task = 50000
    n_eval_per_task = 1000
    max_train = train_range[1] - 1  # 17520

    test_ranges = OrderedDict([
        ('in_range',      (1, 17521)),       # Years 0-2 (training)
        ('ood_2y_5y',     (17521, 43801)),    # Years 2-5
        ('ood_5y_10y',    (43801, 87601)),    # Years 5-10
        ('ood_10y_50y',   (87601, 438001)),   # Years 10-50
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

    # Label distribution check for multi-class tasks
    for task_name in ['hourofday', 'dayofweek', 'timeslot', 'season']:
        tid = TASK_ID[task_name]
        mask = (train_data[0] == tid)
        task_labels = train_data[2][mask]
        unique, counts = torch.unique(task_labels, return_counts=True)
        print(f"    {task_name:15s} label range: {unique.min().item()}-{unique.max().item()}, "
              f"n_unique={len(unique)}")

    # Verify temporal correctness with known timestamps
    print("\n  Temporal sanity check (t=0 is padding, t=1+ are valid):")
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
            max_num=max_train
        )
        all_results[cfg_name] = result

    return all_results, test_datasets


# ============================================================
# PART B: SCALING TEST (d_model=256)
# ============================================================

def run_part_b():
    print("\n" + "=" * 70)
    print("PHASE 12 PART B: SCALING TEST (d_model=256)")
    print("2 configs at larger model scale")
    print("=" * 70)

    d_model = 256
    nhead = 8
    num_layers = 6
    epochs = 30
    batch_size = 512
    lr = 3e-4

    train_range = (1, 17521)
    n_train_per_task = 50000
    n_eval_per_task = 1000
    max_train = train_range[1] - 1

    test_ranges = OrderedDict([
        ('in_range',      (1, 17521)),
        ('ood_2y_5y',     (17521, 43801)),
        ('ood_5y_10y',    (43801, 87601)),
        ('ood_10y_50y',   (87601, 438001)),
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
        ('spirnor_256', EMBED_CONFIGS['spirnor_temporal']),
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
            max_num=max_train
        )
        all_results[cfg_name] = result

    return all_results


# ============================================================
# RESULTS SUMMARY
# ============================================================

def print_summary(a_results, b_results):
    range_keys = ['in_range', 'ood_2y_5y', 'ood_5y_10y', 'ood_10y_50y']
    range_labels = ['In-Range', '2-5 Yrs', '5-10 Yrs', '10-50 Yrs']

    print("\n" + "=" * 70)
    print("PHASE 12 RESULTS SUMMARY")
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

    # ---- Per-Family Analysis ----
    families = {
        'A': {'name': 'Direct Temporal', 'tasks': ['hourofday', 'dayofweek']},
        'B': {'name': 'Hierarchical Composition', 'tasks': ['isweekend', 'timeslot', 'isbusinesshour', 'season']},
        'C': {'name': 'Multi-Timestamp', 'tasks': ['samedow', 'hourdiffmod12']},
    }

    print("\n" + "-" * 70)
    print("PER-FAMILY ANALYSIS (2-5 Years OOD)")
    print("-" * 70)

    for fam_id, fam in families.items():
        print(f"\n  Family {fam_id}: {fam['name']}")
        for cfg_name, res in a_results.items():
            vals = []
            task_strs = []
            for t in fam['tasks']:
                v = res.get('ood_2y_5y', {}).get(t, 0)
                vals.append(v)
                task_strs.append(f"{TASKS[t]['name']}={v:.3f}")
            avg = sum(vals) / len(vals) if vals else 0
            print(f"    {cfg_name:20s}: avg={avg:.4f}  [{', '.join(task_strs)}]")

    # ---- Full Per-Task Breakdown ----
    print("\n" + "-" * 70)
    print("FULL PER-TASK BREAKDOWN AT 2-5 YEARS OOD")
    print("-" * 70)

    task_header = f"  {'Config':20s}"
    for task_name in TASK_LIST:
        task_header += f" | {TASKS[task_name]['name']:>12s}"
    print(task_header)
    print("  " + "-" * (len(task_header) - 2))

    for cfg_name, res in a_results.items():
        line = f"  {cfg_name:20s}"
        for task_name in TASK_LIST:
            val = res.get('ood_2y_5y', {}).get(task_name, 0)
            line += f" | {val:.4f}      "
        print(line)

    # ---- SPIRNOR vs Learned advantage ----
    if 'spirnor_temporal' in a_results and 'learned' in a_results:
        print("\n" + "-" * 70)
        print("SPIRNOR ADVANTAGE AT 2-5 YEARS OOD")
        print("-" * 70)

        for task_name in TASK_LIST:
            s_val = a_results['spirnor_temporal'].get('ood_2y_5y', {}).get(task_name, 0)
            l_val = a_results['learned'].get('ood_2y_5y', {}).get(task_name, 0)
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
            ('spirnor_temporal', 'spirnor_256'),
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

    for cfg_name, res in a_results.items():
        in_acc = res.get('in_range', {}).get('overall', 0)
        ood_vals = [res.get(k, {}).get('overall', 0) for k in range_keys[1:]]
        avg_ood = sum(ood_vals) / len(ood_vals) if ood_vals else 0
        print(f"  {cfg_name:20s}: in_range={in_acc:.4f}, avg_OOD={avg_ood:.4f}")

    # Temporal CRT validation
    if 'spirnor_temporal' in a_results:
        hod_ood = a_results['spirnor_temporal'].get('ood_10y_50y', {}).get('hourofday', 0)
        print(f"\n  Temporal CRT: HourOfDay @ 10-50y = {hod_ood:.4f} "
              f"({'PASS' if hod_ood > 0.95 else 'INVESTIGATE'})")

    # Hierarchical composition check
    if 'spirnor_temporal' in a_results:
        print(f"\n  Hierarchical Composition (SPIRNOR @ 2-5y OOD):")
        comp_tasks = ['hourofday', 'dayofweek', 'isweekend', 'timeslot', 'isbusinesshour', 'season']
        for t in comp_tasks:
            v = a_results['spirnor_temporal'].get('ood_2y_5y', {}).get(t, 0)
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
        'phase': 'Phase 12: SPIRNOR for Temporal/Periodic Data',
        'config': {
            'part_a': {
                'd_model': 128, 'nhead': 4, 'num_layers': 6, 'epochs': 30,
                'train_range': [1, 17521], 'n_train_per_task': 50000,
                'n_eval_per_task': 1000, 'n_tasks': N_TASKS,
                'tasks': list(TASK_LIST),
                'spirnor_constants': TEMPORAL_NAMES,
                'temporal_periods': TEMPORAL_PERIODS,
            },
            'part_b': {
                'd_model': 256, 'nhead': 8, 'num_layers': 6, 'epochs': 30,
                'train_range': [1, 17521], 'n_train_per_task': 50000,
                'n_eval_per_task': 1000,
            },
        },
        'task_families': {
            'A_DirectTemporal': ['hourofday', 'dayofweek'],
            'B_Hierarchical': ['isweekend', 'timeslot', 'isbusinesshour', 'season'],
            'C_MultiTimestamp': ['samedow', 'hourdiffmod12'],
        },
        'results_part_a': clean(a_results),
        'results_part_b': clean(b_results) if b_results else {},
    }

    save_path = 'SPIRNOR_AI_PHASE12_RESULTS.json'
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
    print("Phase 12 complete!")
