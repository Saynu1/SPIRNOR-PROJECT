#!/usr/bin/env python3
"""
SPIRNOR AI Phase 10: Autoregressive Arithmetic — SPIRNOR in LLM Context

Can SPIRNOR embeddings help a GPT-style transformer generalize integer
arithmetic to numbers larger than seen during training?

This is the critical test for SPIRNOR's practical LLM applicability:
- Autoregressive generation (like real LLMs, not classification)
- Integer addition is THE canonical LLM failure mode
- Tests both numeric value augmentation and position encoding

Part A: Addition (6 embedding configs, d_model=128)
  1. baseline      — Learned char embeddings + sinusoidal position
  2. spirnor_aug   — + SPIRNOR(number_value) for input digit tokens
  3. learned_aug   — + nn.Embedding(number_value) for input digit tokens
  4. std_rope      — Learned char embeddings + standard RoPE
  5. spirnor_rope  — Learned char embeddings + SPIRNOR-frequency RoPE
  6. spirnor_full  — spirnor_aug + spirnor_rope combined

Part B: Multiplication (same 6 configs)

Key comparisons:
  - spirnor_aug vs learned_aug: isolates STRUCTURAL benefit of SPIRNOR
    (learned_aug has same info in-range but loses it OOD)
  - spirnor_rope vs std_rope: tests prime-rational position frequencies
  - spirnor_full: tests combined value + position encoding

Training: 200K examples per operation
  Addition: operands 1-999 (1-3 digits)
  Multiplication: operands 2-99 (1-2 digits)
OOD: progressively larger operands (4-digit, 5-digit, 6-digit)
Eval: exact match + per-digit accuracy + length accuracy

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
# VOCABULARY & CONSTANTS
# ============================================================

CHARS = list('0123456789+*=')
SPECIAL = ['<PAD>', '<BOS>', '<EOS>']
VOCAB = CHARS + SPECIAL
CHAR2IDX = {ch: i for i, ch in enumerate(VOCAB)}
IDX2CHAR = {i: ch for ch, i in CHAR2IDX.items()}

PAD_IDX = CHAR2IDX['<PAD>']
BOS_IDX = CHAR2IDX['<BOS>']
EOS_IDX = CHAR2IDX['<EOS>']
EQ_IDX = CHAR2IDX['=']
VOCAB_SIZE = len(VOCAB)

# SPIRNOR constants: rational 2pi/p for first 5 primes (Phase 8 winner)
PHI = (1 + math.sqrt(5)) / 2
TWO_PI = 2 * math.pi
SPIRNOR_PRIMES = [2, 3, 5, 7, 11]
SPIRNOR_CONSTANTS = [TWO_PI / p for p in SPIRNOR_PRIMES]
SPIRNOR_NAMES = [f'2pi/{p}' for p in SPIRNOR_PRIMES]

MAX_SEQ_LEN = 30  # BOS + 6-digit + op + 6-digit + = + 7-digit + EOS + padding

# ============================================================
# DATA GENERATION
# ============================================================

def make_seq_str(a, b, op):
    """Create 'a op b = c' string."""
    if op == '+':
        c = a + b
    elif op == '*':
        c = a * b
    return f"{a}{op}{b}={c}"


def tokenize(seq_str):
    """Convert string to token index list with BOS/EOS."""
    return [BOS_IDX] + [CHAR2IDX[ch] for ch in seq_str] + [EOS_IDX]


def compute_input_values(seq_str):
    """For each character, return the full number value it belongs to.
    Only annotate digits BEFORE '=' (input operands, not answer).
    Returns list of ints, same length as seq_str.
    """
    eq_pos = seq_str.index('=')
    values = []
    i = 0
    while i < len(seq_str):
        if i < eq_pos and seq_str[i].isdigit():
            # Start of a number
            j = i
            while j < eq_pos and seq_str[j].isdigit():
                j += 1
            num_val = int(seq_str[i:j])
            values.extend([num_val] * (j - i))
            i = j
        else:
            values.append(0)
            i += 1
    return values


def generate_data(op, n_examples, lo, hi, rng):
    """Generate tokenized arithmetic data.

    Returns:
        tokens:    [N, MAX_SEQ_LEN] token indices
        values:    [N, MAX_SEQ_LEN] number values per position
        loss_mask: [N, MAX_SEQ_LEN] 1 for answer+EOS positions
    """
    all_tokens = []
    all_values = []

    for _ in range(n_examples):
        a = int(rng.randint(lo, hi + 1))
        b = int(rng.randint(lo, hi + 1))
        seq_str = make_seq_str(a, b, op)
        toks = tokenize(seq_str)
        vals = [0] + compute_input_values(seq_str) + [0]  # BOS=0, EOS=0

        # Pad
        while len(toks) < MAX_SEQ_LEN:
            toks.append(PAD_IDX)
        while len(vals) < MAX_SEQ_LEN:
            vals.append(0)

        all_tokens.append(toks[:MAX_SEQ_LEN])
        all_values.append(vals[:MAX_SEQ_LEN])

    tokens = torch.tensor(all_tokens, dtype=torch.long)
    values = torch.tensor(all_values, dtype=torch.long)

    # Loss mask: 1 for answer tokens + EOS (everything after '=' that isn't PAD)
    eq_mask = (tokens == EQ_IDX)
    after_eq = eq_mask.cumsum(dim=1) > 0
    not_pad = (tokens != PAD_IDX)
    not_eq = (tokens != EQ_IDX)
    loss_mask = (after_eq & not_pad & not_eq).float()

    return tokens, values, loss_mask


# ============================================================
# SPIRNOR VALUE PROJECTION
# ============================================================

class SPIRNORValueProjection(nn.Module):
    """Project SPIRNOR features of a numeric value to d_model.
    Augments character embeddings for digit tokens with modular structure."""

    def __init__(self, d_model, const_values, const_names=None):
        super().__init__()
        self.const_names = const_names or [f'C{i}' for i in range(len(const_values))]
        self.register_buffer('const_vals',
                             torch.tensor(const_values, dtype=torch.float32))
        self.n_consts = len(const_values)
        raw_dim = 7 * self.n_consts
        self.proj = nn.Linear(raw_dim, d_model)

    def forward(self, values):
        """values: [B, S] integer number values. 0 = no augmentation."""
        mask = (values > 0).unsqueeze(-1).float()
        n_vals = values.float().clamp(min=1.0)

        all_feats = []
        for i in range(self.n_consts):
            C = self.const_vals[i]
            r = torch.log(n_vals)
            theta = (C * n_vals) % TWO_PI
            phi_angle = (PHI * n_vals) % TWO_PI
            x = r * torch.sin(theta) * torch.cos(phi_angle)
            y = r * torch.sin(theta) * torch.sin(phi_angle)
            z = r * torch.cos(theta)
            feats = torch.stack([x, y, z,
                                 torch.sin(theta), torch.cos(theta),
                                 torch.sin(phi_angle), torch.cos(phi_angle)], dim=-1)
            all_feats.append(feats)

        raw = torch.cat(all_feats, dim=-1)
        embedded = self.proj(raw)
        return embedded * mask  # zero where value == 0


class LearnedValueProjection(nn.Module):
    """Learned embedding of numeric values. Values > max_val get zero."""

    def __init__(self, max_val, d_model):
        super().__init__()
        self.max_val = max_val
        self.embed = nn.Embedding(max_val + 1, d_model, padding_idx=0)

    def forward(self, values):
        """values: [B, S]. OOD values > max_val → index 0 (zero embedding)."""
        clamped = values.clone()
        clamped[values > self.max_val] = 0
        return self.embed(clamped)


# ============================================================
# POSITION ENCODINGS
# ============================================================

class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_SEQ_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


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
    def __init__(self, d_head, max_len=MAX_SEQ_LEN, base=10000):
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
    """SPIRNOR-frequency RoPE: uses 2pi/p for primes p as frequencies.
    Creates exact mod-p positional encodings."""

    def __init__(self, d_head, max_len=MAX_SEQ_LEN):
        super().__init__()
        n_freqs = d_head // 2
        primes = self._gen_primes(n_freqs)
        freqs = torch.tensor([TWO_PI / p for p in primes], dtype=torch.float32)
        t = torch.arange(max_len, dtype=torch.float32)
        angles = torch.outer(t, freqs)
        self.register_buffer('cos_cached', angles.cos())
        self.register_buffer('sin_cached', angles.sin())

    @staticmethod
    def _gen_primes(n):
        primes = []
        c = 2
        while len(primes) < n:
            if all(c % p != 0 for p in primes):
                primes.append(c)
            c += 1
        return primes

    def forward(self, x, seq_len):
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        return apply_rotary(x, cos, sin)


# ============================================================
# GPT MODEL (Decoder-Only Transformer)
# ============================================================

class CausalSelfAttention(nn.Module):
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
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                             dropout_p=drop_p)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(out)


class DecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, nhead, dropout)
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


class ArithmeticGPT(nn.Module):
    """GPT-style decoder-only transformer for arithmetic.

    embed_config controls the embedding strategy:
      'baseline':     char_embed + sinusoidal_pos
      'spirnor_aug':  char_embed + SPIRNOR(value) + sinusoidal_pos
      'learned_aug':  char_embed + learned(value) + sinusoidal_pos
      'std_rope':     char_embed + standard RoPE
      'spirnor_rope': char_embed + SPIRNOR-freq RoPE
      'spirnor_full': char_embed + SPIRNOR(value) + SPIRNOR-freq RoPE
    """

    def __init__(self, d_model, nhead, num_layers, d_ff,
                 embed_config='baseline', max_val=999, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embed_config = embed_config

        # Character embedding (shared with output head via weight tying)
        self.char_embed = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_IDX)

        # Value augmentation
        self.value_aug = None
        if embed_config in ('spirnor_aug', 'spirnor_full'):
            self.value_aug = SPIRNORValueProjection(
                d_model, SPIRNOR_CONSTANTS, SPIRNOR_NAMES)
        elif embed_config == 'learned_aug':
            self.value_aug = LearnedValueProjection(max_val, d_model)

        # Position encoding
        use_rope = embed_config in ('std_rope', 'spirnor_rope', 'spirnor_full')
        self.pos_enc = None
        self.rotary = None

        if use_rope:
            d_head = d_model // nhead
            if embed_config == 'std_rope':
                self.rotary = StandardRoPE(d_head)
            else:
                self.rotary = SPIRNORRoPE(d_head)
        else:
            self.pos_enc = SinusoidalPositionEncoding(d_model)

        # Transformer decoder
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Output head (weight-tied with char_embed)
        self.lm_head = nn.Linear(d_model, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.char_embed.weight

    def forward(self, tokens, values=None):
        """
        tokens: [B, S] token indices
        values: [B, S] numeric values (0 = no augmentation)
        Returns: [B, S, VOCAB_SIZE] logits
        """
        x = self.char_embed(tokens)

        if self.value_aug is not None and values is not None:
            x = x + self.value_aug(values)

        if self.pos_enc is not None:
            x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x, self.rotary)

        x = self.final_norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, prefix_tokens, prefix_values=None, max_new=15):
        """Autoregressive generation from prefix."""
        self.eval()
        tokens = prefix_tokens.clone()
        values = prefix_values.clone() if prefix_values is not None else None
        generated = []

        for _ in range(max_new):
            logits = self.forward(tokens, values)
            next_token = logits[0, -1].argmax().item()
            if next_token in (EOS_IDX, PAD_IDX):
                break
            generated.append(next_token)
            nt = torch.tensor([[next_token]], device=tokens.device)
            tokens = torch.cat([tokens, nt], dim=1)
            if values is not None:
                nv = torch.zeros((1, 1), dtype=torch.long, device=values.device)
                values = torch.cat([values, nv], dim=1)

        return generated


# ============================================================
# TRAINING
# ============================================================

def train_model(model, tokens, values, loss_mask,
                epochs=30, batch_size=512, lr=3e-4):
    """Train with teacher forcing. Loss only on answer + EOS tokens."""
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Teacher forcing split
    input_tokens = tokens[:, :-1]
    target_tokens = tokens[:, 1:]
    input_values = values[:, :-1]
    target_mask = loss_mask[:, 1:]  # shift to align with targets

    n = len(tokens)
    start = time.time()

    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        total_correct = 0
        total_masked = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            b_in = input_tokens[idx].to(device)
            b_tgt = target_tokens[idx].to(device)
            b_val = input_values[idx].to(device)
            b_mask = target_mask[idx].to(device)

            optimizer.zero_grad()
            logits = model(b_in, b_val)

            logits_flat = logits.reshape(-1, VOCAB_SIZE)
            target_flat = b_tgt.reshape(-1)
            mask_flat = b_mask.reshape(-1)

            loss = F.cross_entropy(logits_flat, target_flat, reduction='none')
            loss = (loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * mask_flat.sum().item()
            preds = logits_flat.argmax(dim=-1)
            total_correct += ((preds == target_flat) * mask_flat).sum().item()
            total_masked += mask_flat.sum().item()

        scheduler.step()
        avg_loss = total_loss / max(total_masked, 1)
        avg_acc = total_correct / max(total_masked, 1)
        elapsed = time.time() - start

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}: loss={avg_loss:.4f}, "
                  f"token_acc={avg_acc:.4f}, time={elapsed:.0f}s")

    return time.time() - start


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, op, lo, hi, n_eval=1000, seed=42):
    """Evaluate autoregressive generation accuracy."""
    model.eval()
    rng = np.random.RandomState(seed)

    exact_correct = 0
    length_correct = 0
    total_digits = 0
    correct_digits = 0

    for _ in range(n_eval):
        a = int(rng.randint(lo, hi + 1))
        b = int(rng.randint(lo, hi + 1))

        if op == '+':
            answer = str(a + b)
        elif op == '*':
            answer = str(a * b)

        # Build prefix: "a op b ="
        prefix_str = f"{a}{op}{b}="
        prefix_tokens = [BOS_IDX] + [CHAR2IDX[ch] for ch in prefix_str]
        prefix_values = [0] + compute_input_values(prefix_str)

        pt = torch.tensor([prefix_tokens], dtype=torch.long, device=device)
        pv = torch.tensor([prefix_values], dtype=torch.long, device=device)

        max_answer_len = len(answer) + 4
        gen_indices = model.generate(pt, pv, max_new=max_answer_len)
        gen_str = ''.join(IDX2CHAR.get(idx, '?') for idx in gen_indices)

        # Exact match
        if gen_str == answer:
            exact_correct += 1

        # Length match
        if len(gen_str) == len(answer):
            length_correct += 1

        # Per-digit (left-aligned)
        for i in range(max(len(answer), len(gen_str))):
            total_digits += 1
            if i < len(answer) and i < len(gen_str):
                if gen_str[i] == answer[i]:
                    correct_digits += 1

    return {
        'exact_match': exact_correct / n_eval,
        'length_acc': length_correct / n_eval,
        'per_digit': correct_digits / max(total_digits, 1),
    }


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_experiment():
    print("=" * 70)
    print("PHASE 10: SPIRNOR FOR AUTOREGRESSIVE ARITHMETIC")
    print("=" * 70)
    print(f"Vocabulary: {VOCAB_SIZE} tokens")
    print(f"SPIRNOR constants: {SPIRNOR_NAMES}")
    print(f"Max sequence length: {MAX_SEQ_LEN}")

    # ---- HYPERPARAMETERS ----
    D_MODEL = 128
    NHEAD = 4
    NUM_LAYERS = 6
    D_FF = 512
    EPOCHS = 30
    BATCH = 512
    LR = 3e-4

    N_TRAIN = 200000
    N_EVAL = 1000

    ADD_LO, ADD_HI = 1, 999
    MUL_LO, MUL_HI = 2, 99

    CONFIGS = [
        'baseline',
        'spirnor_aug',
        'learned_aug',
        'std_rope',
        'spirnor_rope',
        'spirnor_full',
    ]

    ADD_EVAL = {
        'in_range':  (1, 999),
        '4_digit':   (1000, 9999),
        '5_digit':   (10000, 99999),
        '6_digit':   (100000, 999999),
    }

    MUL_EVAL = {
        'in_range':     (2, 99),
        '3_digit':      (100, 999),
        '3_digit_mix':  (2, 999),      # one small, one large
    }

    results = {
        'config': {
            'd_model': D_MODEL, 'nhead': NHEAD, 'num_layers': NUM_LAYERS,
            'd_ff': D_FF, 'epochs': EPOCHS, 'batch': BATCH, 'lr': LR,
            'n_train': N_TRAIN, 'n_eval': N_EVAL,
            'spirnor_constants': SPIRNOR_NAMES,
        }
    }

    # ---- GENERATE DATA ----
    rng = np.random.RandomState(42)

    print(f"\nGenerating addition training data: {N_TRAIN} examples, "
          f"operands [{ADD_LO}, {ADD_HI}]...")
    add_tok, add_val, add_mask = generate_data('+', N_TRAIN, ADD_LO, ADD_HI, rng)
    print(f"  Shape: {add_tok.shape}")

    print(f"Generating multiplication training data: {N_TRAIN} examples, "
          f"operands [{MUL_LO}, {MUL_HI}]...")
    mul_tok, mul_val, mul_mask = generate_data('*', N_TRAIN, MUL_LO, MUL_HI, rng)
    print(f"  Shape: {mul_tok.shape}")

    # ---- PART A: ADDITION ----
    print("\n" + "=" * 70)
    print("PART A: ADDITION (train 1-999, OOD to 6-digit)")
    print("=" * 70)

    for cfg in CONFIGS:
        print(f"\n--- Config: {cfg} ---")
        model = ArithmeticGPT(
            D_MODEL, NHEAD, NUM_LAYERS, D_FF,
            embed_config=cfg, max_val=ADD_HI, dropout=0.1)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        t = train_model(model, add_tok, add_val, add_mask,
                        epochs=EPOCHS, batch_size=BATCH, lr=LR)
        print(f"  Training time: {t:.0f}s")

        cfg_results = {'params': n_params, 'train_time': round(t, 1)}
        for rname, (lo, hi) in ADD_EVAL.items():
            res = evaluate_model(model, '+', lo, hi, n_eval=N_EVAL)
            cfg_results[rname] = res
            print(f"  {rname:14s}: exact={res['exact_match']:.3f}, "
                  f"digit={res['per_digit']:.3f}, len={res['length_acc']:.3f}")

        results[f'add_{cfg}'] = cfg_results
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- PART B: MULTIPLICATION ----
    print("\n" + "=" * 70)
    print("PART B: MULTIPLICATION (train 2-99, OOD to 3-digit)")
    print("=" * 70)

    for cfg in CONFIGS:
        print(f"\n--- Config: {cfg} ---")
        model = ArithmeticGPT(
            D_MODEL, NHEAD, NUM_LAYERS, D_FF,
            embed_config=cfg, max_val=MUL_HI, dropout=0.1)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        t = train_model(model, mul_tok, mul_val, mul_mask,
                        epochs=EPOCHS, batch_size=BATCH, lr=LR)
        print(f"  Training time: {t:.0f}s")

        cfg_results = {'params': n_params, 'train_time': round(t, 1)}
        for rname, (lo, hi) in MUL_EVAL.items():
            res = evaluate_model(model, '*', lo, hi, n_eval=N_EVAL)
            cfg_results[rname] = res
            print(f"  {rname:14s}: exact={res['exact_match']:.3f}, "
                  f"digit={res['per_digit']:.3f}, len={res['length_acc']:.3f}")

        results[f'mul_{cfg}'] = cfg_results
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- SUMMARY TABLES ----
    print("\n" + "=" * 70)
    print("SUMMARY: ADDITION — Exact Match Accuracy")
    print("=" * 70)

    header = f"{'Config':16s}"
    for rname in ADD_EVAL:
        header += f" | {rname:>12s}"
    print(header)
    print("-" * len(header))
    for cfg in CONFIGS:
        key = f'add_{cfg}'
        row = f"{cfg:16s}"
        for rname in ADD_EVAL:
            v = results[key].get(rname, {}).get('exact_match', 0)
            row += f" |       {v:.3f}"
        print(row)

    print(f"\n{'SUMMARY: MULTIPLICATION — Exact Match Accuracy':^70}")
    print("=" * 70)

    header = f"{'Config':16s}"
    for rname in MUL_EVAL:
        header += f" | {rname:>14s}"
    print(header)
    print("-" * len(header))
    for cfg in CONFIGS:
        key = f'mul_{cfg}'
        row = f"{cfg:16s}"
        for rname in MUL_EVAL:
            v = results[key].get(rname, {}).get('exact_match', 0)
            row += f" |         {v:.3f}"
        print(row)

    # ---- SPIRNOR ADVANTAGE ----
    print("\n" + "=" * 70)
    print("SPIRNOR ADVANTAGE ANALYSIS")
    print("=" * 70)

    # spirnor_aug vs learned_aug (value augmentation)
    print("\n1. Value Augmentation: spirnor_aug vs learned_aug")
    for rname in ADD_EVAL:
        s = results['add_spirnor_aug'].get(rname, {}).get('exact_match', 0)
        l = results['add_learned_aug'].get(rname, {}).get('exact_match', 0)
        diff = s - l
        print(f"   ADD {rname:12s}: spirnor={s:.3f}, learned={l:.3f}, "
              f"delta={diff:+.3f}")

    # spirnor_rope vs std_rope (position encoding)
    print("\n2. Position Encoding: spirnor_rope vs std_rope")
    for rname in ADD_EVAL:
        s = results['add_spirnor_rope'].get(rname, {}).get('exact_match', 0)
        l = results['add_std_rope'].get(rname, {}).get('exact_match', 0)
        diff = s - l
        print(f"   ADD {rname:12s}: spirnor={s:.3f}, standard={l:.3f}, "
              f"delta={diff:+.3f}")

    # Best config vs baseline
    print("\n3. Best SPIRNOR vs Baseline")
    for rname in ADD_EVAL:
        best_val = 0
        best_cfg = ''
        for cfg in CONFIGS:
            v = results[f'add_{cfg}'].get(rname, {}).get('exact_match', 0)
            if 'spirnor' in cfg and v > best_val:
                best_val = v
                best_cfg = cfg
        base_val = results['add_baseline'].get(rname, {}).get('exact_match', 0)
        diff = best_val - base_val
        print(f"   ADD {rname:12s}: best_spirnor({best_cfg})={best_val:.3f}, "
              f"baseline={base_val:.3f}, delta={diff:+.3f}")

    # ---- SAVE ----
    save_path = 'SPIRNOR_AI_PHASE10_RESULTS.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")

    return results


if __name__ == '__main__':
    run_experiment()
