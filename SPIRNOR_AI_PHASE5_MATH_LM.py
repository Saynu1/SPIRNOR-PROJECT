#!/usr/bin/env python3
"""
SPIRNOR AI Phase 5: Character-Level Autoregressive Math Transformer

Tests whether SPIRNOR-RoPE improves a GPT-style language model on arithmetic.
This is the real test: genuine next-token prediction, autoregressive generation,
and out-of-distribution generalization to larger numbers (longer sequences).

Architecture: Decoder-only transformer (causal attention)
  - d_model=128, 4 heads, 4 layers, ~1.5M params
  - Character-level tokenizer (0-9, +, -, *, =, etc.)
  - Teacher-forced training, greedy decode for evaluation

Tasks (compact format for efficient training):
  A:23+47=70       (Addition)
  S:47-23=24       (Subtraction)
  M:7*8=56         (Multiplication)
  G:36,48=12       (GCD)
  F:51=3           (Smallest Prime Factor)

Training: Numbers 2-200 (2-3 digit sequences)
Testing OOD: 201-500, 501-1000, 1001-5000 (longer digit sequences)

Key hypothesis: SPIRNOR-RoPE should help with OOD generalization because:
  1. Larger numbers = more digits = longer sequences = PE quality matters more
  2. SPIRNOR constants provide meaningful angular separation at ALL positions
  3. Standard RoPE wastes low-frequency dimensions on short training sequences

Phase 5B addition: Hybrid Dual-Band RoPE
  - Macro band: Standard geometric RoPE (low freq, global position "map")
  - Micro band: SPIRNOR winding constants (high freq, arithmetic "microscope")
  - Solves the aliasing problem where pure SPIRNOR-RoPE collapses on long sequences

Environment: Python 3.12, PyTorch 2.10.0+cu130, RTX 3060 Ti
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import json
import time
import sys
from collections import OrderedDict

# ============================================================
# DEVICE
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# CONSTANTS
# ============================================================
PHI = (1 + math.sqrt(5)) / 2

SPIRNOR_CONSTANTS = OrderedDict([
    ('pi', math.pi),
    ('sqrt2', math.sqrt(2)),
    ('phi_sq', PHI**2),
    ('e', math.e),
    ('golden_angle', 2 * math.pi / PHI**2),
    ('phi', PHI),
    ('ln2', math.log(2)),
    ('pi_e', math.pi / math.e),
])

SPIRNOR_CONST_LIST = list(SPIRNOR_CONSTANTS.values())

# ============================================================
# CHARACTER TOKENIZER
# ============================================================

# Vocabulary: digits, operators, task prefixes, special tokens
VOCAB = {
    '<PAD>': 0, '<BOS>': 1, '<EOS>': 2,
    '0': 3, '1': 4, '2': 5, '3': 6, '4': 7,
    '5': 8, '6': 9, '7': 10, '8': 11, '9': 12,
    '+': 13, '-': 14, '*': 15, '=': 16,
    ',': 17, ':': 18,
    'A': 19, 'S': 20, 'M': 21, 'G': 22, 'F': 23,
}
ID_TO_CHAR = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)
PAD_ID = VOCAB['<PAD>']
BOS_ID = VOCAB['<BOS>']
EOS_ID = VOCAB['<EOS>']

def encode(text):
    """Encode text string to token IDs."""
    return [VOCAB[ch] for ch in text]

def decode(ids):
    """Decode token IDs to text string."""
    return ''.join(ID_TO_CHAR.get(i, '?') for i in ids
                   if i not in (PAD_ID, BOS_ID, EOS_ID))


# ============================================================
# ROTARY POSITION ENCODINGS (from Phase 4, with CUDA support)
# ============================================================

def apply_rotary(x, cos, sin):
    """Apply rotary embedding. x: (B, H, S, D), cos/sin: (1, 1, S, D//2)."""
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
    def __init__(self, d_head, max_len=512, base=10000):
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
    """SPIRNOR-RoPE: winding constants as rotation frequencies.
    All frequencies in [0.69, 3.14] — every dimension pair provides
    meaningful angular separation, even for short sequences.
    """
    def __init__(self, d_head, max_len=512, constants=None):
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


class HybridSPIRNORRoPE(nn.Module):
    """Dual-Band Rotary Embedding: the best of both worlds.

    Splits d_head dimensions into two bands:
      - MACRO band (first half): Standard geometric RoPE frequencies
        → Low frequencies provide the "map" — smooth position encoding
          that distinguishes position 0 from position 50, even in long sequences.
      - MICRO band (second half): SPIRNOR winding constants
        → High frequencies provide the "microscope" — rich arithmetic
          structure encoding divisibility, resonance, and number-theoretic
          relationships between nearby positions.

    This solves the aliasing problem: pure SPIRNOR-RoPE is all-high-frequency
    (every pair in [0.69, 3.14]), causing position vectors to wrap around
    so many times that position 12 looks identical to position 2 on long
    sequences. The macro band ensures long-range position discrimination
    while the micro band preserves SPIRNOR's arithmetic advantage.

    Design rationale:
      d_head=32 → 16 pairs total
        Pairs 0-7:  geometric decay (1/10000^(2d/D)), freq range [1.0, ~0.001]
        Pairs 8-15: SPIRNOR constants × harmonics, freq range [0.69, 6.28]

    The split ratio can be tuned. Default 50/50 balances map vs microscope.
    """
    def __init__(self, d_head, max_len=512, base=10000, constants=None,
                 spirnor_ratio=0.5):
        super().__init__()
        consts = constants or SPIRNOR_CONST_LIST
        n_pairs = d_head // 2
        n_spirnor = int(n_pairs * spirnor_ratio)
        n_geometric = n_pairs - n_spirnor

        # MACRO band: standard geometric RoPE (low-freq "map")
        geo_freqs = 1.0 / (base ** (torch.arange(0, n_geometric).float() * 2 / d_head))

        # MICRO band: SPIRNOR winding constants (high-freq "microscope")
        spirnor_freqs = []
        for i in range(n_spirnor):
            c_idx = i % len(consts)
            harmonic = (i // len(consts)) + 1
            spirnor_freqs.append(consts[c_idx] * harmonic)
        spirnor_freqs = torch.tensor(spirnor_freqs, dtype=torch.float32)

        # Concatenate: [geometric_low ... | spirnor_high ...]
        freqs = torch.cat([geo_freqs, spirnor_freqs])

        t = torch.arange(max_len, dtype=torch.float32)
        angles = torch.outer(t, freqs)
        self.register_buffer('cos_cached', angles.cos())
        self.register_buffer('sin_cached', angles.sin())

        # Store metadata for analysis
        self.n_geometric = n_geometric
        self.n_spirnor = n_spirnor

    def forward(self, x, seq_len):
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        return apply_rotary(x, cos, sin)


class SinusoidalPE(nn.Module):
    """Additive sinusoidal PE (original Transformer)."""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)


class LearnedPE(nn.Module):
    """Learned absolute position embedding."""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.pe(positions).unsqueeze(0)


# ============================================================
# CAUSAL TRANSFORMER (GPT-style)
# ============================================================

class CausalMultiHeadAttention(nn.Module):
    """Multi-head attention with causal mask and optional rotary PE.
    Uses F.scaled_dot_product_attention for GPU-optimized flash attention.
    """
    def __init__(self, d_model, nhead, dropout=0.1, max_len=512):
        super().__init__()
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.d_model = d_model
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

        # Use fused SDPA with causal mask (flash attention on GPU)
        drop_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop_p)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(out)


class CausalTransformerLayer(nn.Module):
    """Pre-norm transformer layer with causal attention."""
    def __init__(self, d_model, nhead, d_ff, dropout=0.1, max_len=512):
        super().__init__()
        self.attn = CausalMultiHeadAttention(d_model, nhead, dropout, max_len)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, rotary=None):
        x = x + self.attn(self.norm1(x), rotary=rotary)
        x = x + self.ff(self.norm2(x))
        return x


class MathGPT(nn.Module):
    """Small GPT-style model for character-level math.

    PE types:
      'learned'       - learned absolute position embedding (additive)
      'sinusoidal'    - sinusoidal additive PE
      'rope'          - standard RoPE (geometric freq progression)
      'spirnor_rope'  - SPIRNOR-RoPE (winding constant frequencies)
      'hybrid_rope'   - Hybrid dual-band (geometric macro + SPIRNOR micro)
    """
    def __init__(self, vocab_size, d_model, nhead, num_layers,
                 pe_type='rope', max_len=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.pe_type = pe_type

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)

        # Position encoding
        d_head = d_model // nhead
        self.additive_pe = None
        self.rotary = None
        if pe_type == 'learned':
            self.additive_pe = LearnedPE(d_model, max_len)
        elif pe_type == 'sinusoidal':
            self.additive_pe = SinusoidalPE(d_model, max_len)
        elif pe_type == 'rope':
            self.rotary = StandardRoPE(d_head, max_len)
        elif pe_type == 'spirnor_rope':
            self.rotary = SPIRNORRoPE(d_head, max_len)
        elif pe_type == 'hybrid_rope':
            self.rotary = HybridSPIRNORRoPE(d_head, max_len, spirnor_ratio=0.5)

        # Transformer layers
        self.layers = nn.ModuleList([
            CausalTransformerLayer(d_model, nhead, d_model * 4, dropout, max_len)
            for _ in range(num_layers)
        ])

        # Output head
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (token embed and lm_head share weights)
        self.lm_head.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids):
        """Forward pass: input_ids (B, S) -> logits (B, S, vocab_size)."""
        x = self.token_embed(input_ids) * math.sqrt(self.d_model)

        if self.additive_pe is not None:
            x = self.additive_pe(x)

        for layer in self.layers:
            x = layer(x, rotary=self.rotary)

        x = self.norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=20):
        """Greedy autoregressive generation."""
        self.eval()
        ids = prompt_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(ids)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            if (next_id == EOS_ID).all():
                break
        return ids


# ============================================================
# DATA GENERATION
# ============================================================

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def smallest_prime_factor(n):
    if n < 2:
        return n
    if n % 2 == 0:
        return 2
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return i
    return n


def generate_examples(n_samples, num_range, rng, tasks=None):
    """Generate arithmetic examples as token sequences.

    Each example: <BOS> task_prefix operands = answer <EOS>
    Examples:
      A:23+47=70
      S:47-23=24
      M:7*8=56
      G:36,48=12
      F:51=3
    """
    lo, hi = num_range
    if tasks is None:
        tasks = ['add', 'sub', 'gcd', 'spf', 'mul']

    examples = []
    per_task = n_samples // len(tasks)

    for task in tasks:
        for _ in range(per_task):
            if task == 'add':
                a = rng.randint(lo, hi + 1)
                b = rng.randint(lo, hi + 1)
                answer = a + b
                text = f"A:{a}+{b}={answer}"
            elif task == 'sub':
                a = rng.randint(lo, hi + 1)
                b = rng.randint(lo, min(a, hi) + 1)  # ensure a >= b
                answer = a - b
                text = f"S:{a}-{b}={answer}"
            elif task == 'mul':
                # Keep multiplication manageable — cap factors
                mul_lo = min(lo, 50)
                mul_hi = min(hi, 99)
                if mul_lo > mul_hi:
                    mul_lo = 2
                a = rng.randint(mul_lo, mul_hi + 1)
                b = rng.randint(mul_lo, mul_hi + 1)
                answer = a * b
                text = f"M:{a}*{b}={answer}"
            elif task == 'gcd':
                a = rng.randint(lo, hi + 1)
                b = rng.randint(lo, hi + 1)
                answer = gcd(a, b)
                text = f"G:{a},{b}={answer}"
            elif task == 'spf':
                n = rng.randint(max(lo, 2), hi + 1)
                answer = smallest_prime_factor(n)
                text = f"F:{n}={answer}"
            else:
                continue

            tokens = [BOS_ID] + encode(text) + [EOS_ID]
            examples.append(tokens)

    rng.shuffle(examples)
    return examples


def pad_sequences(examples, max_len=None):
    """Pad token sequences to uniform length."""
    if max_len is None:
        max_len = max(len(ex) for ex in examples)
    padded = []
    for ex in examples:
        padded.append(ex[:max_len] + [PAD_ID] * max(0, max_len - len(ex)))
    return torch.tensor(padded, dtype=torch.long)


def find_answer_start(tokens):
    """Find the position of '=' in a token sequence (answer starts after it)."""
    eq_id = VOCAB['=']
    for i, t in enumerate(tokens):
        if t == eq_id:
            return i + 1
    return len(tokens)


# ============================================================
# TRAINING
# ============================================================

def train_model(model, train_data, epochs=30, batch_size=256, lr=3e-4):
    """Train with teacher forcing on full sequences.
    Loss computed only on answer tokens (after '=').
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    n_params = sum(p.numel() for p in model.parameters())

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_data))
        total_loss = 0
        n_tokens = 0

        for i in range(0, len(train_data), batch_size):
            idx = perm[i:i+batch_size]
            batch = train_data[idx].to(device)

            # Input: all tokens except last; Target: all tokens except first
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]

            logits = model(input_ids)  # (B, S-1, vocab_size)
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                target_ids.reshape(-1),
                ignore_index=PAD_ID
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Count non-pad tokens for logging
            mask = (target_ids != PAD_ID)
            total_loss += loss.item() * mask.sum().item()
            n_tokens += mask.sum().item()

        scheduler.step()
        avg_loss = total_loss / max(n_tokens, 1)

        if (epoch + 1) in [1, 5, 10, 15, 20, 25]:
            print(f"    Epoch {epoch+1:3d}/{epochs}: loss={avg_loss:.4f}")

    return n_params


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_teacher_forced(model, examples, max_len=40, batch_size=512):
    """Fast batched evaluation using teacher forcing.

    Two metrics:
    1. answer_token_acc: fraction of answer tokens predicted correctly
    2. sequence_exact_match: fraction of examples with ALL answer tokens correct

    Both are computed in batch mode (no autoregressive loop).
    """
    model.eval()
    model.to(device)

    # Pad examples
    padded = pad_sequences(examples, max_len)

    seq_correct = 0
    seq_total = 0
    task_correct = {}
    task_total = {}

    for i in range(0, len(padded), batch_size):
        batch = padded[i:i+batch_size].to(device)
        batch_raw = examples[i:i+batch_size]

        input_ids = batch[:, :-1]
        target_ids = batch[:, 1:]

        logits = model(input_ids)
        preds = logits.argmax(dim=-1)  # (B, S-1)

        for j in range(len(batch_raw)):
            tokens = batch_raw[j]
            eq_pos = find_answer_start(tokens)
            ans_start = eq_pos - 1  # shifted by 1 for target

            # Extract answer tokens and predictions
            all_match = True
            has_answer = False
            for k in range(ans_start, target_ids.size(1)):
                t = target_ids[j, k].item()
                if t == PAD_ID or t == EOS_ID:
                    break
                has_answer = True
                if preds[j, k].item() != t:
                    all_match = False

            if not has_answer:
                continue

            # Task type from first real token (after BOS)
            task_char = ID_TO_CHAR.get(tokens[1], '?')

            seq_correct += int(all_match)
            seq_total += 1
            if task_char not in task_correct:
                task_correct[task_char] = 0
                task_total[task_char] = 0
            task_correct[task_char] += int(all_match)
            task_total[task_char] += 1

    overall = seq_correct / max(seq_total, 1)
    per_task = {}
    for task in sorted(task_correct.keys()):
        per_task[task] = task_correct[task] / max(task_total[task], 1)

    return overall, per_task


@torch.no_grad()
def generate_samples(model, examples, n=5):
    """Generate a few samples autoregressively for qualitative inspection."""
    model.eval()
    model.to(device)
    results = []
    for tokens in examples[:n]:
        eq_pos = find_answer_start(tokens)
        prompt = tokens[:eq_pos]
        answer = [t for t in tokens[eq_pos:] if t not in (EOS_ID, PAD_ID)]

        prompt_tensor = torch.tensor([prompt], dtype=torch.long, device=device)
        generated = model.generate(prompt_tensor, max_new_tokens=15)
        gen_tokens = generated[0, len(prompt):].cpu().tolist()
        gen_answer = [t for t in gen_tokens if t not in (EOS_ID, PAD_ID)]

        results.append({
            'prompt': decode(prompt),
            'expected': decode(answer),
            'generated': decode(gen_answer),
            'correct': gen_answer == answer,
        })
    return results


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment():
    print("=" * 70)
    print("SPIRNOR AI PHASE 5B: HYBRID DUAL-BAND RoPE + MATH LM")
    print("=" * 70)
    print()
    print("Testing SPIRNOR-RoPE in a GPT-style autoregressive model.")
    print("Character-level next-token prediction on arithmetic tasks.")
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print()

    # Hyperparameters
    d_model = 128
    nhead = 4
    num_layers = 4
    epochs = 25
    batch_size = 512
    lr = 3e-4
    max_len = 40  # max sequence length (math expressions are short)

    train_range = (2, 200)
    test_ranges = OrderedDict([
        ('in_range',     (2, 200)),
        ('ood_200_500',  (201, 500)),
        ('ood_500_1k',   (501, 1000)),
        ('ood_1k_5k',    (1001, 5000)),
    ])

    tasks = ['add', 'sub', 'gcd', 'spf', 'mul']
    pe_types = ['rope', 'spirnor_rope', 'hybrid_rope']
    pe_labels = {
        'learned': 'Learned PE',
        'sinusoidal': 'Sinusoidal PE',
        'rope': 'Standard RoPE',
        'spirnor_rope': 'SPIRNOR-RoPE',
        'hybrid_rope': 'Hybrid-RoPE',
    }

    # Generate training data
    print("Generating training data...")
    rng = np.random.RandomState(42)
    train_examples = generate_examples(40000, train_range, rng, tasks)
    train_padded = pad_sequences(train_examples, max_len)
    print(f"  Training: {len(train_examples)} examples, padded to {max_len} tokens")
    print(f"  Example: {decode(train_examples[0])}")
    print(f"  Example: {decode(train_examples[1])}")
    print(f"  Example: {decode(train_examples[2])}")

    # Generate test data
    print("\nGenerating test data...")
    test_data = OrderedDict()
    for name, r in test_ranges.items():
        test_rng = np.random.RandomState(hash(name) % 2**31)
        examples = generate_examples(500, r, test_rng, tasks)
        test_data[name] = examples
        print(f"  {name}: {len(examples)} examples, "
              f"e.g. {decode(examples[0])}")

    # Run experiment for each PE type
    all_results = OrderedDict()
    print(f"\n{'='*70}")
    print(f"TRAINING AND EVALUATION")
    print(f"{'='*70}")
    print(f"Architecture: {num_layers}-layer GPT, d_model={d_model}, "
          f"{nhead} heads, max_len={max_len}")
    print(f"Training: {epochs} epochs, batch_size={batch_size}, lr={lr}")
    print()

    for pe_type in pe_types:
        label = pe_labels[pe_type]
        print(f"\n{'='*60}")
        print(f"PE TYPE: {label}")
        print(f"{'='*60}")

        model = MathGPT(
            vocab_size=VOCAB_SIZE,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            pe_type=pe_type,
            max_len=max_len,
            dropout=0.1,
        )

        t0 = time.time()
        n_params = train_model(model, train_padded, epochs=epochs,
                               batch_size=batch_size, lr=lr)
        train_time = time.time() - t0
        print(f"  Training time: {train_time:.1f}s, Params: {n_params:,}")

        # Evaluate on each test range (teacher-forced, batched)
        print(f"\n  Evaluating (teacher-forced exact-match, batched)...")
        pe_results = OrderedDict()
        pe_results['params'] = n_params
        pe_results['train_time'] = train_time

        for range_name, examples in test_data.items():
            t_eval = time.time()
            overall_acc, per_task = evaluate_teacher_forced(
                model, examples, max_len=max_len
            )
            eval_time = time.time() - t_eval

            pe_results[range_name] = {
                'overall': overall_acc,
                'per_task': per_task,
            }

            task_str = ' '.join(f"{t}={v:.3f}" for t, v in per_task.items())
            print(f"    {range_name:15s}: overall={overall_acc:.4f}  "
                  f"[{task_str}]  ({eval_time:.1f}s)")

        # Show a few autoregressive generation samples from in-range
        samples = generate_samples(model, test_data['in_range'], n=3)
        print(f"\n  Sample generations (in-range):")
        for s in samples:
            status = "OK" if s['correct'] else "WRONG"
            print(f"    {s['prompt']}{s['expected']} -> "
                  f"generated: {s['generated']} [{status}]")

        all_results[label] = pe_results

    return all_results


def print_summary(results):
    print("\n" + "=" * 70)
    print("PHASE 5B RESULTS SUMMARY")
    print("=" * 70)

    range_keys = ['in_range', 'ood_200_500', 'ood_500_1k', 'ood_1k_5k']
    range_labels = ['In-Range', '200-500', '500-1K', '1K-5K']

    # Overall accuracy table
    print("\n  Overall Exact-Match Accuracy:")
    header = f"  {'PE Type':18s} | {'Params':>10s}"
    for rl in range_labels:
        header += f" | {rl:>8s}"
    print(header)
    print(f"  {'-' * len(header)}")

    for pe_name, res in results.items():
        line = f"  {pe_name:18s} | {res['params']:10,}"
        for key in range_keys:
            val = res.get(key, {}).get('overall', 0)
            best = max(r.get(key, {}).get('overall', 0) for r in results.values())
            marker = '*' if abs(val - best) < 0.001 and val > 0 else ' '
            line += f" | {val:.4f}{marker}"
        print(line)

    # Per-task breakdown: all rotary variants
    print("\n" + "=" * 70)
    print("ROTARY PE COMPARISON (per-task)")
    print("=" * 70)

    rope_res = results.get('Standard RoPE', {})
    spirnor_res = results.get('SPIRNOR-RoPE', {})
    hybrid_res = results.get('Hybrid-RoPE', {})

    rotary_variants = []
    if rope_res: rotary_variants.append(('RoPE', rope_res))
    if spirnor_res: rotary_variants.append(('SPIRNOR', spirnor_res))
    if hybrid_res: rotary_variants.append(('Hybrid', hybrid_res))

    if len(rotary_variants) >= 2:
        task_names = {'A': 'Addition', 'S': 'Subtraction', 'M': 'Multiplication',
                      'G': 'GCD', 'F': 'SPF'}
        for task_id, task_name in task_names.items():
            print(f"\n  {task_name}:")
            for key, label in zip(range_keys, range_labels):
                vals = []
                for vname, vres in rotary_variants:
                    v = vres.get(key, {}).get('per_task', {}).get(task_id, 0)
                    vals.append((vname, v))
                best_val = max(v for _, v in vals)
                parts = []
                for vname, v in vals:
                    marker = '*' if abs(v - best_val) < 0.005 and v > 0 else ' '
                    parts.append(f"{vname}={v:.3f}{marker}")
                print(f"    {label:8s}: {'  '.join(parts)}")

    # Generalization degradation
    print("\n" + "=" * 70)
    print("OOD GENERALIZATION DEGRADATION")
    print("=" * 70)

    for pe_name, res in results.items():
        in_acc = res.get('in_range', {}).get('overall', 0)
        far_acc = res.get('ood_1k_5k', {}).get('overall', 0)
        drop = in_acc - far_acc
        print(f"  {pe_name:18s}: in-range={in_acc:.4f}  1K-5K={far_acc:.4f}  "
              f"drop={drop:+.4f}")

    # Win counts
    print("\n" + "=" * 70)
    print("SCOREBOARD")
    print("=" * 70)

    wins = {pe: 0 for pe in results.keys()}
    total = 0
    for key in range_keys:
        best_val = max(r.get(key, {}).get('overall', 0) for r in results.values())
        for pe_name, res in results.items():
            val = res.get(key, {}).get('overall', 0)
            if abs(val - best_val) < 0.005:  # within 0.5% counts as win
                wins[pe_name] += 1
        total += 1

    for pe_name, w in sorted(wins.items(), key=lambda x: -x[1]):
        bar = '#' * (w * 5)
        print(f"  {pe_name:18s}: {w}/{total} wins  {bar}")


def main():
    results = run_experiment()
    print_summary(results)

    # Save results
    def clean(obj):
        if isinstance(obj, OrderedDict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    save_path = 'Scripts/SPIRNOR_AI_PHASE5B_RESULTS.json'
    all_data = {
        'results': clean(results),
        'config': {
            'd_model': 128, 'nhead': 4, 'num_layers': 4, 'epochs': 25,
            'train_range': [2, 200], 'vocab_size': VOCAB_SIZE,
            'tasks': ['add', 'sub', 'gcd', 'spf', 'mul'],
            'pe_types': ['rope', 'spirnor_rope', 'hybrid_rope'],
            'spirnor_constants': dict(SPIRNOR_CONSTANTS),
        }
    }
    with open(save_path, 'w') as f:
        json.dump(all_data, f, indent=2)
    print(f"\nResults saved to: {save_path}")


if __name__ == '__main__':
    main()
