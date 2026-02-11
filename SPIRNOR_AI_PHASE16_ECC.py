#!/usr/bin/env python3
"""
SPIRNOR AI Phase 16: Elliptic Curve Cryptography

Extends Phase 15's scalar modular arithmetic to 2D coordinate arithmetic
on elliptic curves. Tests whether SPIRNOR's exact mod-p encoding works
for the algebraic group operations underlying modern ECC.

3 Curves:
  E7a: y^2 = x^3 + 5      (mod 7),  order 7  (prime, cyclic)
  E7b: y^2 = x^3 + x      (mod 7),  order 8  (composite, 2^3)
  E11: y^2 = x^3 + x + 6  (mod 11), order 13 (prime, cyclic)

8 Tasks across 5 Families:
  1. IsOnCurve7:       (x,y) -> {0,1}            [2 classes]
  2. IsOnCurve11:      (x,y) -> {0,1}            [2 classes]
  3. PointAddX7:       (x1,y1,x2,y2) -> x3       [8 classes]
  4. ScalarMulX7:      (k) -> x-coord of kG      [8 classes]
  5. ScalarMulIdx11:   (k) -> point index of kG   [13 classes]
  6. ECDLP7:           (Qx,Qy) -> k              [6 classes]
  7. ECDLP11:          (Qx,Qy) -> k              [12 classes]
  8. ECDH7:            (a,b) -> point idx of abG  [7 classes]

Theory:
  - All ECC ops reduce to modular arithmetic in F_p
  - SPIRNOR encodes x mod 7, y mod 7 (and mod 11) exactly
  - Point addition formula is pure mod-p arithmetic over 4 coordinates
  - ECDLP is a lookup from (Qx mod p, Qy mod p) -> k

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

MAX_SEQ = 4  # Maximum number inputs per example (PointAddX7 uses 4)

# ============================================================
# ELLIPTIC CURVE MATH
# ============================================================

def mod_inv(a, p):
    """Modular inverse via Fermat's little theorem: a^(p-2) mod p."""
    return pow(a % p, p - 2, p)


def ec_add(P, Q, a_coeff, p):
    """Add points P and Q on y^2 = x^3 + a*x + b over F_p.
    Points are (x, y) tuples or None for point at infinity O."""
    if P is None:
        return Q
    if Q is None:
        return P
    px, py = P
    qx, qy = Q
    if px == qx and (py + qy) % p == 0:
        return None  # P + (-P) = O
    if px == qx and py == qy:
        # Point doubling
        num = (3 * px * px + a_coeff) % p
        den = (2 * py) % p
        if den == 0:
            return None
        lam = (num * mod_inv(den, p)) % p
    else:
        # Point addition
        num = (qy - py) % p
        den = (qx - px) % p
        if den == 0:
            return None
        lam = (num * mod_inv(den, p)) % p
    x3 = (lam * lam - px - qx) % p
    y3 = (lam * (px - x3) - py) % p
    return (x3, y3)


def ec_scalar_mul(k, P, a_coeff, p):
    """Compute k*P using double-and-add."""
    if k == 0 or P is None:
        return None
    k = k % 1000000007  # Safety for very large k (use with group order)
    result = None
    addend = P
    while k > 0:
        if k & 1:
            result = ec_add(result, addend, a_coeff, p)
        addend = ec_add(addend, addend, a_coeff, p)
        k >>= 1
    return result


# ============================================================
# CURVE DEFINITIONS (all verified by hand)
# ============================================================

# E7a: y^2 = x^3 + 5 (mod 7), a=0, b=5, order=7
E7A = {'a': 0, 'b': 5, 'p': 7, 'order': 7, 'G': (3, 2)}

# Precompute E7a scalar multiplication table
E7A_TABLE = {}  # k -> point (or None for O)
for _k in range(E7A['order'] + 1):
    E7A_TABLE[_k] = ec_scalar_mul(_k, E7A['G'], E7A['a'], E7A['p'])

# E7a point -> scalar index lookup (for ECDLP)
E7A_DLOG = {}  # (x, y) -> k (1-indexed)
for _k in range(1, E7A['order']):
    pt = E7A_TABLE[_k]
    if pt is not None:
        E7A_DLOG[pt] = _k

# E7a point index mapping: 0=O, 1..6 = points in order of k*G
E7A_POINT_IDX = {None: 0}
for _k in range(1, E7A['order']):
    E7A_POINT_IDX[E7A_TABLE[_k]] = _k


# E7b: y^2 = x^3 + x (mod 7), a=1, b=0, order=8
E7B = {'a': 1, 'b': 0, 'p': 7, 'order': 8, 'G': (3, 3)}

# Precompute E7b scalar multiplication table
E7B_TABLE = {}
for _k in range(E7B['order'] + 1):
    E7B_TABLE[_k] = ec_scalar_mul(_k, E7B['G'], E7B['a'], E7B['p'])

# E7b point index mapping: 0=O, 1..7 = points in order of k*G
E7B_POINT_IDX = {None: 0}
for _k in range(1, E7B['order']):
    E7B_POINT_IDX[E7B_TABLE[_k]] = _k

# E7b: All valid curve points (for IsOnCurve positive sampling)
E7B_POINTS = []
for _x in range(E7B['p']):
    rhs = (_x**3 + E7B['a'] * _x + E7B['b']) % E7B['p']
    for _y in range(E7B['p']):
        if (_y * _y) % E7B['p'] == rhs:
            E7B_POINTS.append((_x, _y))


# E11: y^2 = x^3 + x + 6 (mod 11), a=1, b=6, order=13
E11 = {'a': 1, 'b': 6, 'p': 11, 'order': 13, 'G': (2, 4)}

# Precompute E11 scalar multiplication table
E11_TABLE = {}
for _k in range(E11['order'] + 1):
    E11_TABLE[_k] = ec_scalar_mul(_k, E11['G'], E11['a'], E11['p'])

# E11 point -> scalar index lookup (for ECDLP)
E11_DLOG = {}
for _k in range(1, E11['order']):
    pt = E11_TABLE[_k]
    if pt is not None:
        E11_DLOG[pt] = _k

# E11 point index mapping: 0=O, 1..12 = points in order of k*G
E11_POINT_IDX = {None: 0}
for _k in range(1, E11['order']):
    E11_POINT_IDX[E11_TABLE[_k]] = _k

# E11: All valid curve points (for IsOnCurve positive sampling)
E11_POINTS = []
for _x in range(E11['p']):
    rhs = (_x**3 + E11['a'] * _x + E11['b']) % E11['p']
    for _y in range(E11['p']):
        if (_y * _y) % E11['p'] == rhs:
            E11_POINTS.append((_x, _y))


# Verify curve data at import time
def _verify_curves():
    # E7a: order 7
    assert E7A_TABLE[0] is None, "0*G should be O"
    assert E7A_TABLE[7] is None, "7*G should be O (order 7)"
    assert E7A_TABLE[1] == (3, 2), f"1*G should be (3,2), got {E7A_TABLE[1]}"
    assert E7A_TABLE[2] == (5, 2), f"2*G should be (5,2), got {E7A_TABLE[2]}"
    assert E7A_TABLE[3] == (6, 5), f"3*G should be (6,5), got {E7A_TABLE[3]}"
    assert len(E7A_DLOG) == 6, f"E7a should have 6 non-O points, got {len(E7A_DLOG)}"

    # E7b: order 8
    assert E7B_TABLE[0] is None, "0*G should be O"
    assert E7B_TABLE[8] is None, "8*G should be O (order 8)"
    assert E7B_TABLE[1] == (3, 3), f"1*G should be (3,3), got {E7B_TABLE[1]}"
    assert E7B_TABLE[4] == (0, 0), f"4*G should be (0,0), got {E7B_TABLE[4]}"
    assert len(E7B_POINTS) == 7, f"E7b should have 7 finite points, got {len(E7B_POINTS)}"

    # E11: order 13
    assert E11_TABLE[0] is None, "0*G should be O"
    assert E11_TABLE[13] is None, "13*G should be O (order 13)"
    assert E11_TABLE[1] == (2, 4), f"1*G should be (2,4), got {E11_TABLE[1]}"
    assert len(E11_DLOG) == 12, f"E11 should have 12 non-O points, got {len(E11_DLOG)}"
    assert len(E11_POINTS) == 12, f"E11 should have 12 finite points, got {len(E11_POINTS)}"

    print("All curve verifications passed!")
    print(f"  E7a: {len(E7A_DLOG)+1} points (order {E7A['order']}), G={E7A['G']}")
    print(f"  E7b: {len(E7B_POINTS)+1} points (order {E7B['order']}), G={E7B['G']}")
    print(f"  E11: {len(E11_DLOG)+1} points (order {E11['order']}), G={E11['G']}")

_verify_curves()


# ============================================================
# TASK DEFINITIONS (8 tasks across 5 ECC families)
# ============================================================

TASKS = OrderedDict([
    # Family 1: Point Validation
    ('isoncurve7',     {'n_inputs': 2, 'n_classes': 2,  'name': 'IsOnCurve7',     'family': '1_Validation'}),
    ('isoncurve11',    {'n_inputs': 2, 'n_classes': 2,  'name': 'IsOnCurve11',    'family': '1_Validation'}),
    # Family 2: Point Arithmetic
    ('pointaddx7',     {'n_inputs': 4, 'n_classes': 8,  'name': 'PointAddX7',     'family': '2_PointArith'}),
    # Family 3: Scalar Multiplication
    ('scalarmulx7',    {'n_inputs': 1, 'n_classes': 8,  'name': 'ScalarMulX7',    'family': '3_ScalarMul'}),
    ('scalarmulidx11', {'n_inputs': 1, 'n_classes': 13, 'name': 'ScalarMulIdx11', 'family': '3_ScalarMul'}),
    # Family 4: EC Discrete Logarithm
    ('ecdlp7',         {'n_inputs': 2, 'n_classes': 6,  'name': 'ECDLP7',         'family': '4_ECDLP'}),
    ('ecdlp11',        {'n_inputs': 2, 'n_classes': 12, 'name': 'ECDLP11',        'family': '4_ECDLP'}),
    # Family 5: EC Diffie-Hellman
    ('ecdh7',          {'n_inputs': 2, 'n_classes': 7,  'name': 'ECDH7',          'family': '5_ECDH'}),
])

TASK_LIST = list(TASKS.keys())
TASK_ID = {name: i for i, name in enumerate(TASK_LIST)}
N_TASKS = len(TASK_LIST)


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
                 task_configs, max_num=2000, max_len=6,
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
# ELEVATION HELPER
# ============================================================

def elevate_coord(coord_val, p, lo, hi, rng):
    """Elevate a small F_p coordinate to a large integer in [lo, hi]
    such that result % p == coord_val."""
    lo_mult = max(0, math.ceil((lo - coord_val) / p))
    hi_mult = (hi - coord_val) // p
    if hi_mult < lo_mult:
        # Fallback: just use coord_val + p * lo_mult
        return coord_val + p * lo_mult
    mult = int(rng.randint(lo_mult, hi_mult + 1))
    return coord_val + p * mult


# ============================================================
# DATA GENERATION
# ============================================================

def generate_dataset(n_per_task, num_range, rng):
    """Generate training/eval data for all 8 Phase 16 ECC tasks."""
    lo, hi = num_range
    all_task_ids = []
    all_numbers = []
    all_labels = []

    for task_name in TASK_LIST:
        task_id = TASK_ID[task_name]

        if task_name == 'isoncurve7':
            # (x, y) -> {0=not on E7b, 1=on E7b}
            # Balanced 50/50 sampling
            n_pos = n_per_task // 2
            n_neg = n_per_task - n_pos

            # Positive examples: pick valid curve points, elevate
            for _ in range(n_pos):
                px, py = E7B_POINTS[int(rng.randint(0, len(E7B_POINTS)))]
                x_big = elevate_coord(px, 7, lo, hi, rng)
                y_big = elevate_coord(py, 7, lo, hi, rng)
                all_task_ids.append(task_id)
                all_numbers.append([x_big, y_big, 0, 0])
                all_labels.append(1)

            # Negative examples: random, reject if on curve
            count = 0
            attempts = 0
            while count < n_neg and attempts < n_neg * 5:
                attempts += 1
                x_big = int(rng.randint(lo, hi + 1))
                y_big = int(rng.randint(lo, hi + 1))
                xm, ym = x_big % 7, y_big % 7
                rhs = (xm**3 + xm) % 7  # E7b: a=1, b=0
                if (ym * ym) % 7 == rhs:
                    continue  # On curve, reject
                all_task_ids.append(task_id)
                all_numbers.append([x_big, y_big, 0, 0])
                all_labels.append(0)
                count += 1

        elif task_name == 'isoncurve11':
            # (x, y) -> {0=not on E11, 1=on E11}
            n_pos = n_per_task // 2
            n_neg = n_per_task - n_pos

            for _ in range(n_pos):
                px, py = E11_POINTS[int(rng.randint(0, len(E11_POINTS)))]
                x_big = elevate_coord(px, 11, lo, hi, rng)
                y_big = elevate_coord(py, 11, lo, hi, rng)
                all_task_ids.append(task_id)
                all_numbers.append([x_big, y_big, 0, 0])
                all_labels.append(1)

            count = 0
            attempts = 0
            while count < n_neg and attempts < n_neg * 5:
                attempts += 1
                x_big = int(rng.randint(lo, hi + 1))
                y_big = int(rng.randint(lo, hi + 1))
                xm, ym = x_big % 11, y_big % 11
                rhs = (xm**3 + xm + 6) % 11  # E11: a=1, b=6
                if (ym * ym) % 11 == rhs:
                    continue
                all_task_ids.append(task_id)
                all_numbers.append([x_big, y_big, 0, 0])
                all_labels.append(0)
                count += 1

        elif task_name == 'pointaddx7':
            # (x1, y1, x2, y2) -> x-coord of P1+P2 on E7b (or 7 for O)
            count = 0
            attempts = 0
            while count < n_per_task and attempts < n_per_task * 5:
                attempts += 1
                k1 = int(rng.randint(lo, hi + 1))
                k2 = int(rng.randint(lo, hi + 1))
                k1m = k1 % E7B['order']
                k2m = k2 % E7B['order']
                P1 = E7B_TABLE[k1m]
                P2 = E7B_TABLE[k2m]
                if P1 is None or P2 is None:
                    continue  # Skip if either is O
                # Compute P1 + P2
                P3 = ec_add(P1, P2, E7B['a'], E7B['p'])
                label = P3[0] if P3 is not None else 7  # x-coord or 7 for O

                # Elevate coordinates to target range
                x1_big = elevate_coord(P1[0], 7, lo, hi, rng)
                y1_big = elevate_coord(P1[1], 7, lo, hi, rng)
                x2_big = elevate_coord(P2[0], 7, lo, hi, rng)
                y2_big = elevate_coord(P2[1], 7, lo, hi, rng)

                all_task_ids.append(task_id)
                all_numbers.append([x1_big, y1_big, x2_big, y2_big])
                all_labels.append(label)
                count += 1

        elif task_name == 'scalarmulx7':
            # (k) -> x-coord of kG on E7a (or 7 for O)
            for _ in range(n_per_task):
                k = int(rng.randint(lo, hi + 1))
                km = k % E7A['order']
                P = E7A_TABLE[km]
                label = P[0] if P is not None else 7
                all_task_ids.append(task_id)
                all_numbers.append([k, 0, 0, 0])
                all_labels.append(label)

        elif task_name == 'scalarmulidx11':
            # (k) -> point index of kG on E11 (0=O, 1-12 for points)
            for _ in range(n_per_task):
                k = int(rng.randint(lo, hi + 1))
                km = k % E11['order']
                P = E11_TABLE[km]
                label = E11_POINT_IDX[P]
                all_task_ids.append(task_id)
                all_numbers.append([k, 0, 0, 0])
                all_labels.append(label)

        elif task_name == 'ecdlp7':
            # (Qx, Qy) -> k-1 where Q = kG on E7a, k in {1..6}
            # Labels 0-5 for k=1..6. Skip k%7==0 (O has no coords).
            count = 0
            attempts = 0
            while count < n_per_task and attempts < n_per_task * 5:
                attempts += 1
                k = int(rng.randint(lo, hi + 1))
                km = k % E7A['order']
                if km == 0:
                    continue  # O has no coordinates
                Q = E7A_TABLE[km]
                label = km - 1  # 0-indexed: k=1->0, k=2->1, ..., k=6->5
                # Elevate Q coordinates
                qx_big = elevate_coord(Q[0], 7, lo, hi, rng)
                qy_big = elevate_coord(Q[1], 7, lo, hi, rng)
                all_task_ids.append(task_id)
                all_numbers.append([qx_big, qy_big, 0, 0])
                all_labels.append(label)
                count += 1

        elif task_name == 'ecdlp11':
            # (Qx, Qy) -> k-1 where Q = kG on E11, k in {1..12}
            # Labels 0-11 for k=1..12. Skip k%13==0.
            count = 0
            attempts = 0
            while count < n_per_task and attempts < n_per_task * 5:
                attempts += 1
                k = int(rng.randint(lo, hi + 1))
                km = k % E11['order']
                if km == 0:
                    continue
                Q = E11_TABLE[km]
                label = km - 1
                qx_big = elevate_coord(Q[0], 11, lo, hi, rng)
                qy_big = elevate_coord(Q[1], 11, lo, hi, rng)
                all_task_ids.append(task_id)
                all_numbers.append([qx_big, qy_big, 0, 0])
                all_labels.append(label)
                count += 1

        elif task_name == 'ecdh7':
            # (a, b) -> point index of (a*b)G on E7a
            # Point index: 0=O, 1..6 for kG where k=1..6
            for _ in range(n_per_task):
                a = int(rng.randint(lo, hi + 1))
                b = int(rng.randint(lo, hi + 1))
                ab_mod = (a * b) % E7A['order']
                P = E7A_TABLE[ab_mod]
                label = E7A_POINT_IDX[P]
                all_task_ids.append(task_id)
                all_numbers.append([a, b, 0, 0])
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
    ('spirnor_rational5', {
        'type': 'spirnor',
        'desc': 'SPIRNOR with rational_5 {2pi/2,3,5,7,11}',
        'const_values': RATIONAL_5_VALUES,
        'const_names': RATIONAL_5_NAMES,
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
    print("PHASE 16 PART A: ELLIPTIC CURVE CRYPTOGRAPHY (d_model=128)")
    print("2 embedding configs x 8 ECC tasks")
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
        print(f"    {task_name:18s}: {task_counts.get(tid, 0):,} examples")

    # Label distribution check
    for task_name in TASK_LIST:
        tid = TASK_ID[task_name]
        mask = (train_data[0] == tid)
        task_labels = train_data[2][mask]
        unique, counts = torch.unique(task_labels, return_counts=True)
        n_classes = TASKS[task_name]['n_classes']
        print(f"    {task_name:18s} labels: {unique.min().item()}-{unique.max().item()}, "
              f"n_unique={len(unique)}/{n_classes}")

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
    print("PHASE 16 PART B: SCALING TEST (d_model=256)")
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
    print("PHASE 16 RESULTS SUMMARY")
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
    families = OrderedDict([
        ('1_Validation', {'name': 'Point Validation',        'tasks': ['isoncurve7', 'isoncurve11']}),
        ('2_PointArith', {'name': 'Point Arithmetic',        'tasks': ['pointaddx7']}),
        ('3_ScalarMul',  {'name': 'Scalar Multiplication',   'tasks': ['scalarmulx7', 'scalarmulidx11']}),
        ('4_ECDLP',      {'name': 'EC Discrete Logarithm',   'tasks': ['ecdlp7', 'ecdlp11']}),
        ('5_ECDH',       {'name': 'EC Diffie-Hellman',       'tasks': ['ecdh7']}),
    ])

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
        task_header += f" | {TASKS[task_name]['name']:>14s}"
    print(task_header)
    print("  " + "-" * (len(task_header) - 2))

    for cfg_name, res in a_results.items():
        line = f"  {cfg_name:20s}"
        for task_name in TASK_LIST:
            val = res.get('ood_2k_5k', {}).get(task_name, 0)
            line += f" | {val:.4f}        "
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
            print(f"  {TASKS[task_name]['name']:14s}: SPIRNOR={s_val:.4f}, "
                  f"Learned={l_val:.4f}, diff={diff:+.4f}, "
                  f"ratio={ratio:.1f}x")

    # ---- Scale invariance per task ----
    if 'spirnor_rational5' in a_results:
        print("\n" + "-" * 70)
        print("SPIRNOR SCALE INVARIANCE (per-task across OOD ranges)")
        print("-" * 70)

        ood_keys = ['ood_2k_5k', 'ood_5k_20k', 'ood_20k_100k', 'ood_100k_500k']

        for task_name in TASK_LIST:
            vals = [a_results['spirnor_rational5'].get(k, {}).get(task_name, 0) for k in ood_keys]
            val_str = ', '.join(f"{v:.3f}" for v in vals)
            drop = vals[0] - vals[-1] if vals else 0
            print(f"  {TASKS[task_name]['name']:14s}: [{val_str}] drop={drop:+.3f}")

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

    for cfg_name, res in a_results.items():
        in_acc = res.get('in_range', {}).get('overall', 0)
        ood_vals = [res.get(k, {}).get('overall', 0) for k in range_keys[1:]]
        avg_ood = sum(ood_vals) / len(ood_vals) if ood_vals else 0
        print(f"  {cfg_name:20s}: in_range={in_acc:.4f}, avg_OOD={avg_ood:.4f}")

    # ECC-specific findings
    if 'spirnor_rational5' in a_results:
        print(f"\n  ECC TASK VALIDATION (SPIRNOR @ 100K-500K):")
        for task_name in TASK_LIST:
            v = a_results['spirnor_rational5'].get('ood_100k_500k', {}).get(task_name, 0)
            status = 'EXACT' if v > 0.99 else 'STRONG' if v > 0.90 else 'MODERATE' if v > 0.70 else 'WEAK'
            bar = '#' * int(v * 30)
            print(f"    {TASKS[task_name]['name']:14s}: {v:.4f} [{status}] {bar}")


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
        'phase': 'Phase 16: Elliptic Curve Cryptography',
        'config': {
            'part_a': {
                'd_model': 128, 'nhead': 4, 'num_layers': 6, 'epochs': 30,
                'train_range': [2, 2000], 'n_train_per_task': 50000,
                'n_eval_per_task': 1000, 'n_tasks': N_TASKS,
                'tasks': list(TASK_LIST),
                'spirnor_constants': RATIONAL_5_NAMES,
                'curves': {
                    'E7a': {'eq': 'y^2 = x^3 + 5 (mod 7)', 'order': 7, 'G': [3, 2]},
                    'E7b': {'eq': 'y^2 = x^3 + x (mod 7)', 'order': 8, 'G': [3, 3]},
                    'E11': {'eq': 'y^2 = x^3 + x + 6 (mod 11)', 'order': 13, 'G': [2, 4]},
                },
            },
            'part_b': {
                'd_model': 256, 'nhead': 8, 'num_layers': 6, 'epochs': 30,
                'train_range': [2, 2000], 'n_train_per_task': 50000,
                'n_eval_per_task': 1000,
            },
        },
        'task_families': {
            '1_Validation': ['isoncurve7', 'isoncurve11'],
            '2_PointArith': ['pointaddx7'],
            '3_ScalarMul': ['scalarmulx7', 'scalarmulidx11'],
            '4_ECDLP': ['ecdlp7', 'ecdlp11'],
            '5_ECDH': ['ecdh7'],
        },
        'results_part_a': clean(a_results),
        'results_part_b': clean(b_results) if b_results else {},
    }

    save_path = 'SPIRNOR_AI_PHASE16_RESULTS.json'
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
    print("Phase 16 complete!")
