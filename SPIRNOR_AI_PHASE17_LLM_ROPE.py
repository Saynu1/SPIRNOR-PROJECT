#!/usr/bin/env python3
"""
SPIRNOR AI Phase 17: SPIRNOR-RoPE in Pretrained Language Models

Tests whether SPIRNOR modifications improve numerical reasoning in a real
pretrained LLM (Pythia-70M). Two intervention types:
  1. RoPE frequency replacement: standard geometric -> prime-based 2pi/p
  2. Value augmentation: inject SPIRNOR features for number tokens

Critical discovery: Pythia-70M uses rotary_pct=0.25 -> only 8 frequency pairs.
First 8 primes {2,3,5,7,11,13,17,19} -> CRT product 9,699,690 covers all OOD.

5 Configurations:
  1. zero_shot        - Pretrained Pythia-70M, no fine-tuning (reference)
  2. baseline_ft      - Standard RoPE, no augmentation, fine-tuned
  3. spirnor_rope_ft  - SPIRNOR RoPE frequencies, fine-tuned
  4. spirnor_value_ft - Standard RoPE + SPIRNOR value augmentation, fine-tuned
  5. spirnor_full_ft  - SPIRNOR RoPE + value augmentation, fine-tuned

6 Math QA Tasks:
  mod7, mod30, gcd, isprime, add, coprime

Train: numbers [2, 2000], ~50K examples, 10 epochs
Eval: in-range + 3 OOD ranges, autoregressive generation, exact match

Self-contained script for Vast.ai cloud deployment.
"""

import subprocess
import sys

def install_packages():
    """Auto-install required packages."""
    packages = [
        'transformers>=4.40.0',
        'accelerate',
        'safetensors',
    ]
    for pkg in packages:
        try:
            name = pkg.split('>=')[0].split('==')[0]
            __import__(name)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                                   pkg, '-q'])

install_packages()

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import json
import time
import re
import functools
import copy
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

TWO_PI = 2 * math.pi
SPIRNOR_ROPE_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19]  # First 8 primes
SPIRNOR_VALUE_PRIMES = [2, 3, 5, 7, 11]  # For value augmentation
N_VALUE_FEATURES = 1 + 2 * len(SPIRNOR_VALUE_PRIMES)  # ln + sin/cos pairs = 11

MODEL_NAME = "EleutherAI/pythia-70m"

# ============================================================
# MATH TASK DEFINITIONS
# ============================================================

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

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

TASKS = OrderedDict([
    ('mod7',    {'gen': lambda a, b: (f"{a} mod 7 = ",    str(a % 7)),
                 'n_inputs': 1}),
    ('mod30',   {'gen': lambda a, b: (f"{a} mod 30 = ",   str(a % 30)),
                 'n_inputs': 1}),
    ('gcd_task',{'gen': lambda a, b: (f"gcd({a},{b}) = ", str(gcd(a, b))),
                 'n_inputs': 2}),
    ('isprime', {'gen': lambda a, b: (f"is_prime({a}) = ",
                                       "yes" if is_prime(a) else "no"),
                 'n_inputs': 1}),
    ('add',     {'gen': lambda a, b: (f"{a} + {b} = ",    str(a + b)),
                 'n_inputs': 2}),
    ('coprime', {'gen': lambda a, b: (f"coprime({a},{b}) = ",
                                       "yes" if gcd(a, b) == 1 else "no"),
                 'n_inputs': 2}),
])

TASK_LIST = list(TASKS.keys())

# ============================================================
# DATA GENERATION
# ============================================================

def generate_examples(task_name, n_examples, lo, hi, rng):
    """Generate (prompt, answer) pairs for a task."""
    gen_fn = TASKS[task_name]['gen']
    examples = []
    for _ in range(n_examples):
        a = int(rng.randint(lo, hi + 1))
        b = int(rng.randint(lo, hi + 1))
        prompt, answer = gen_fn(a, b)
        examples.append((prompt, answer))
    return examples

def generate_all_data(n_per_task, lo, hi, rng):
    """Generate examples for all tasks."""
    all_examples = []
    for task_name in TASK_LIST:
        examples = generate_examples(task_name, n_per_task, lo, hi, rng)
        for prompt, answer in examples:
            all_examples.append((task_name, prompt, answer))
    rng.shuffle(all_examples)
    return all_examples

# ============================================================
# SPIRNOR VALUE AUGMENTATION
# ============================================================

def compute_spirnor_features(n):
    """Compute SPIRNOR features for integer n.
    Returns tensor of shape [11]: [ln(n), sin(2pi/2*n), cos(2pi/2*n), ...]
    """
    n_float = float(max(n, 1))
    feats = [math.log(n_float)]
    for p in SPIRNOR_VALUE_PRIMES:
        angle = (TWO_PI / p) * n_float
        feats.append(math.sin(angle))
        feats.append(math.cos(angle))
    return torch.tensor(feats, dtype=torch.float32)


def extract_numbers_from_text(text):
    """Extract (position, number_value) pairs from text.
    Returns list of (char_start, char_end, int_value)."""
    results = []
    for m in re.finditer(r'\d+', text):
        results.append((m.start(), m.end(), int(m.group())))
    return results


class SPIRNORValueAugmentation(nn.Module):
    """Adds SPIRNOR features to token embeddings for number-containing tokens.
    Projects 11 SPIRNOR features -> d_model and adds to embeddings."""

    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(N_VALUE_FEATURES, d_model, bias=False)
        nn.init.normal_(self.proj.weight, std=0.01)

    def forward(self, hidden_states, spirnor_mask, spirnor_features):
        """
        hidden_states: [B, S, D] token embeddings
        spirnor_mask: [B, S] bool, True for number tokens
        spirnor_features: [B, S, 11] precomputed SPIRNOR features
        """
        aug = self.proj(spirnor_features)  # [B, S, D]
        aug = aug * spirnor_mask.unsqueeze(-1).float()
        return hidden_states + aug


# ============================================================
# MODEL WRAPPER
# ============================================================

class PythiaWithSPIRNOR(nn.Module):
    """Wraps Pythia model with optional SPIRNOR value augmentation."""

    def __init__(self, base_model, tokenizer, use_value_aug=False):
        super().__init__()
        self.model = base_model
        self.tokenizer = tokenizer
        self.use_value_aug = use_value_aug
        self.value_aug = None
        if use_value_aug:
            d_model = base_model.config.hidden_size
            self.value_aug = SPIRNORValueAugmentation(d_model)

    def prepare_spirnor_inputs(self, input_ids):
        """Compute SPIRNOR features for number tokens in the batch."""
        B, S = input_ids.shape
        spirnor_mask = torch.zeros(B, S, dtype=torch.bool, device=input_ids.device)
        spirnor_features = torch.zeros(B, S, N_VALUE_FEATURES,
                                        device=input_ids.device)

        for b in range(B):
            tokens = input_ids[b]
            # Decode each token to find numbers
            # Build char->token mapping
            text = self.tokenizer.decode(tokens, skip_special_tokens=False)
            token_texts = [self.tokenizer.decode([t]) for t in tokens]

            # Map each token to a number if it represents (part of) a number
            char_pos = 0
            token_char_ranges = []
            for t_idx, t_text in enumerate(token_texts):
                start = char_pos
                char_pos += len(t_text)
                token_char_ranges.append((start, char_pos))

            # Extract numbers from full text (only before '=' sign)
            eq_pos = text.find('=')
            if eq_pos == -1:
                eq_pos = len(text)

            numbers = extract_numbers_from_text(text[:eq_pos])

            # For each number, find which tokens it overlaps with
            for num_start, num_end, num_val in numbers:
                for t_idx, (t_start, t_end) in enumerate(token_char_ranges):
                    if t_idx >= S:
                        break
                    # Check overlap
                    if t_start < num_end and t_end > num_start:
                        spirnor_mask[b, t_idx] = True
                        feats = compute_spirnor_features(num_val)
                        spirnor_features[b, t_idx] = feats.to(input_ids.device)

        return spirnor_mask, spirnor_features

    def forward(self, input_ids, attention_mask=None, labels=None):
        if self.use_value_aug and self.value_aug is not None:
            # Get base embeddings
            inputs_embeds = self.model.gpt_neox.embed_in(input_ids)

            # Apply value augmentation
            spirnor_mask, spirnor_features = self.prepare_spirnor_inputs(input_ids)
            inputs_embeds = self.value_aug(inputs_embeds, spirnor_mask,
                                           spirnor_features)

            # Ensure consistent dtype with model (AMP may produce mixed types)
            if inputs_embeds.dtype != self.model.gpt_neox.embed_in.weight.dtype:
                inputs_embeds = inputs_embeds.to(self.model.gpt_neox.embed_in.weight.dtype)

            outputs = self.model(inputs_embeds=inputs_embeds,
                                 attention_mask=attention_mask,
                                 labels=labels)
        else:
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 labels=labels)
        return outputs

    @torch.no_grad()
    def generate_answer(self, prompt_text, max_new_tokens=20):
        """Generate answer autoregressively from prompt."""
        self.eval()
        # CRITICAL: Strip trailing space to match BPE tokenization during training.
        # Training tokenizes "466 mod 7 = 4" where "= " and "4" merge into " 4" token.
        # Eval prompt "466 mod 7 = " with trailing space creates a separate " " token
        # that the model never saw, breaking generation. Stripping aligns eval with training.
        prompt_text = prompt_text.rstrip()
        input_ids = self.tokenizer.encode(prompt_text, return_tensors='pt')
        input_ids = input_ids.to(device)

        generated = input_ids.clone()
        # Use bf16 autocast during generation (matches AMP training)
        use_bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

        for _ in range(max_new_tokens):
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(device.type == 'cuda')):
                if self.use_value_aug and self.value_aug is not None:
                    inputs_embeds = self.model.gpt_neox.embed_in(generated)
                    spirnor_mask, spirnor_features = self.prepare_spirnor_inputs(
                        generated)
                    inputs_embeds = self.value_aug(inputs_embeds, spirnor_mask,
                                                   spirnor_features)
                    outputs = self.model(inputs_embeds=inputs_embeds)
                else:
                    outputs = self.model(generated)

            next_logits = outputs.logits[0, -1, :]
            next_token = next_logits.argmax().unsqueeze(0).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=1)

            # Stop at newline or EOS
            token_text = self.tokenizer.decode(next_token[0])
            if '\n' in token_text or next_token.item() == self.tokenizer.eos_token_id:
                break

        # Extract only generated portion
        gen_ids = generated[0, input_ids.shape[1]:]
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return gen_text.strip()


# ============================================================
# ROPE REPLACEMENT
# ============================================================

def replace_rope_with_spirnor(model):
    """Replace Pythia's standard RoPE frequencies with SPIRNOR prime-based ones.
    Pythia uses rotary_pct=0.25 -> 16 rotary dims -> 8 frequency pairs.
    The rotary_emb is shared at model.gpt_neox.rotary_emb level."""
    rotary_emb = model.gpt_neox.rotary_emb
    n_freqs = rotary_emb.inv_freq.shape[0]

    print(f"  RoPE replacement: {n_freqs} frequency pairs")
    print(f"  Old inv_freq: {rotary_emb.inv_freq.tolist()}")

    # Generate SPIRNOR frequencies: 2pi/prime[i]
    primes = SPIRNOR_ROPE_PRIMES[:n_freqs]
    if len(primes) < n_freqs:
        c = primes[-1] + 1 if primes else 2
        while len(primes) < n_freqs:
            if all(c % p != 0 for p in primes):
                primes.append(c)
            c += 1

    spirnor_inv_freq = torch.tensor(
        [TWO_PI / p for p in primes], dtype=torch.float32,
        device=rotary_emb.inv_freq.device
    )

    print(f"  New inv_freq (SPIRNOR): {spirnor_inv_freq.tolist()}")
    print(f"  Primes used: {primes}")
    print(f"  CRT product: {math.prod(primes):,}")

    # Replace the shared rotary embedding's inv_freq buffer
    rotary_emb.inv_freq = spirnor_inv_freq
    # Also clear any cached cos/sin values so they get recomputed
    if hasattr(rotary_emb, 'cos_cached'):
        delattr(rotary_emb, 'cos_cached')
    if hasattr(rotary_emb, 'sin_cached'):
        delattr(rotary_emb, 'sin_cached')
    if hasattr(rotary_emb, '_cos_cached'):
        delattr(rotary_emb, '_cos_cached')
    if hasattr(rotary_emb, '_sin_cached'):
        delattr(rotary_emb, '_sin_cached')


# ============================================================
# TOKENIZATION & TRAINING
# ============================================================

def tokenize_examples(examples, tokenizer, max_length=48):
    """Tokenize (prompt, answer) pairs for causal LM training.

    Key: BPE tokenization is NOT prefix-stable. Tokenizing "42 mod 7 = "
    alone may produce different tokens than the prefix of "42 mod 7 = 0".
    Fix: strip trailing space from prompt before finding token boundary,
    since "42 mod 7 =" tokenizes identically in both contexts.
    """
    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    n_bad = 0

    for task_name, prompt, answer in examples:
        # Append newline as stop token so model learns when answer ends
        full_text = prompt + answer + '\n'
        encoding = tokenizer(full_text, truncation=True, max_length=max_length,
                             padding='max_length', return_tensors='pt')
        input_ids = encoding['input_ids'][0]
        attention_mask = encoding['attention_mask'][0]

        # Find prompt boundary using stripped prompt (avoids BPE boundary issue)
        prompt_stripped = prompt.rstrip()
        prompt_ids = tokenizer.encode(prompt_stripped, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        # Also mask padding
        labels[attention_mask == 0] = -100

        # Safety check: ensure at least one valid label
        n_valid = (labels != -100).sum().item()
        if n_valid == 0:
            n_bad += 1

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)

    if n_bad > 0:
        print(f"  WARNING: {n_bad}/{len(examples)} examples have all labels=-100!")
    else:
        print(f"  Label check OK: all {len(examples)} examples have valid answer tokens")

    return (torch.stack(all_input_ids),
            torch.stack(all_attention_masks),
            torch.stack(all_labels))


def train_model(wrapped_model, input_ids, attention_mask, labels,
                epochs=10, batch_size=64, lr=2e-5):
    """Fine-tune the wrapped model with mixed precision and warmup."""
    wrapped_model.to(device)
    wrapped_model.train()

    # Optimizer: all parameters (model + value_aug if present)
    trainable = [p for p in wrapped_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01,
                                  eps=1e-6)

    n = len(input_ids)
    steps_per_epoch = (n + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch  # 1 epoch warmup

    # Linear warmup then cosine decay
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision (bf16 if available, else fp16)
    use_bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(not use_bf16 and device.type == 'cuda'))
    print(f"    AMP dtype: {amp_dtype}, GradScaler: {scaler.is_enabled()}")

    start = time.time()
    global_step = 0

    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            b_ids = input_ids[idx].to(device)
            b_mask = attention_mask[idx].to(device)
            b_labels = labels[idx].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=amp_dtype):
                outputs = wrapped_model(b_ids, attention_mask=b_mask,
                                         labels=b_labels)
                loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                n_valid = (b_labels != -100).sum().item()
                print(f"    Bad loss! batch {i//batch_size}, loss={loss.item()}, "
                      f"valid_labels={n_valid}/{b_labels.numel()}")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            total_loss += loss.item()
            n_batches += 1

            # Early sanity check: detect weight corruption
            if global_step <= 3:
                has_nan = any(torch.isnan(p).any() for p in trainable[:5])
                if has_nan:
                    print(f"    WARNING: NaN in weights after step {global_step}!")
                    # Reset model from checkpoint would be ideal, but just log
                    break

        avg_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - start

        if (epoch + 1) in [1, 2, 5, 8, 10]:
            cur_lr = scheduler.get_last_lr()[0]
            print(f"    Epoch {epoch+1:2d}/{epochs}: "
                  f"loss={avg_loss:.4f}, lr={cur_lr:.2e}, time={elapsed:.0f}s")

    return time.time() - start


# ============================================================
# EVALUATION
# ============================================================

def evaluate_config(wrapped_model, task_name, lo, hi, n_eval=1000, seed=42):
    """Evaluate autoregressive generation accuracy for one task/range."""
    wrapped_model.eval()
    rng = np.random.RandomState(seed)
    gen_fn = TASKS[task_name]['gen']

    correct = 0
    for _ in range(n_eval):
        a = int(rng.randint(lo, hi + 1))
        b = int(rng.randint(lo, hi + 1))
        prompt, answer = gen_fn(a, b)

        generated = wrapped_model.generate_answer(prompt, max_new_tokens=15)
        # Clean: take first token/word as answer
        gen_clean = generated.split('\n')[0].split(' ')[0].strip()
        answer_clean = answer.strip()

        if gen_clean == answer_clean:
            correct += 1

    return correct / n_eval


def evaluate_all(wrapped_model, eval_ranges, n_eval_per_task=1000):
    """Evaluate all tasks across all ranges."""
    results = OrderedDict()

    for range_name, (lo, hi) in eval_ranges.items():
        range_results = OrderedDict()
        task_accs = []

        for task_name in TASK_LIST:
            seed = hash((range_name, task_name)) % 2**31
            acc = evaluate_config(wrapped_model, task_name, lo, hi,
                                  n_eval=n_eval_per_task, seed=seed)
            range_results[task_name] = round(acc, 4)
            task_accs.append(acc)

        range_results['overall'] = round(sum(task_accs) / len(task_accs), 4)
        results[range_name] = range_results

        # Print inline
        task_str = ' '.join(f"{t}={range_results[t]:.3f}" for t in TASK_LIST)
        print(f"    {range_name:14s}: overall={range_results['overall']:.4f}  "
              f"[{task_str}]")

    return results


# ============================================================
# EXPERIMENT CONFIGS
# ============================================================

CONFIGS = OrderedDict([
    ('zero_shot', {
        'desc': 'Pretrained Pythia-70M (no fine-tuning)',
        'fine_tune': False,
        'spirnor_rope': False,
        'value_aug': False,
    }),
    ('baseline_ft', {
        'desc': 'Standard RoPE, fine-tuned',
        'fine_tune': True,
        'spirnor_rope': False,
        'value_aug': False,
    }),
    ('spirnor_rope_ft', {
        'desc': 'SPIRNOR RoPE frequencies, fine-tuned',
        'fine_tune': True,
        'spirnor_rope': True,
        'value_aug': False,
    }),
    ('spirnor_value_ft', {
        'desc': 'Standard RoPE + SPIRNOR value augmentation, fine-tuned',
        'fine_tune': True,
        'spirnor_rope': False,
        'value_aug': True,
    }),
    ('spirnor_full_ft', {
        'desc': 'SPIRNOR RoPE + value augmentation, fine-tuned',
        'fine_tune': True,
        'spirnor_rope': True,
        'value_aug': True,
    }),
])


# ============================================================
# SUMMARY & RESULTS
# ============================================================

def print_summary(all_results):
    range_keys = ['in_range', 'ood_2k_5k', 'ood_5k_20k', 'ood_20k_100k']
    range_labels = ['In-Range', '2K-5K', '5K-20K', '20K-100K']

    print("\n" + "=" * 70)
    print("PHASE 17 RESULTS SUMMARY")
    print("=" * 70)

    # Overall accuracy table
    print("\n  Overall Accuracy (avg across 6 tasks)")
    header = f"  {'Config':20s} | {'Params':>10s}"
    for label in range_labels:
        header += f" | {label:>9s}"
    header += f" | {'Avg OOD':>9s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for cfg_name, res in all_results.items():
        n_params = res.get('params', 0)
        line = f"  {cfg_name:20s} | {n_params:10,}"
        ood_vals = []
        for key in range_keys:
            val = res.get(key, {}).get('overall', 0)
            if key != 'in_range':
                ood_vals.append(val)
            line += f" | {val:.4f}   "
        avg_ood = sum(ood_vals) / len(ood_vals) if ood_vals else 0
        line += f" | {avg_ood:.4f}   "
        print(line)

    # Per-task at 2K-5K
    print("\n" + "-" * 70)
    print("PER-TASK ACCURACY AT 2K-5K OOD")
    print("-" * 70)

    task_header = f"  {'Config':20s}"
    for t in TASK_LIST:
        task_header += f" | {t:>10s}"
    print(task_header)
    print("  " + "-" * (len(task_header) - 2))

    for cfg_name, res in all_results.items():
        line = f"  {cfg_name:20s}"
        for t in TASK_LIST:
            val = res.get('ood_2k_5k', {}).get(t, 0)
            line += f" | {val:.4f}    "
        print(line)

    # SPIRNOR advantage analysis
    if 'baseline_ft' in all_results:
        print("\n" + "-" * 70)
        print("SPIRNOR ADVANTAGE vs BASELINE (2K-5K OOD)")
        print("-" * 70)

        base = all_results['baseline_ft']
        for cfg_name in ['spirnor_rope_ft', 'spirnor_value_ft',
                          'spirnor_full_ft']:
            if cfg_name not in all_results:
                continue
            res = all_results[cfg_name]
            print(f"\n  {cfg_name}:")
            for t in TASK_LIST:
                s_val = res.get('ood_2k_5k', {}).get(t, 0)
                b_val = base.get('ood_2k_5k', {}).get(t, 0)
                diff = s_val - b_val
                if b_val > 0:
                    ratio = s_val / b_val
                    print(f"    {t:10s}: {s_val:.4f} vs {b_val:.4f} "
                          f"({diff:+.4f}, {ratio:.1f}x)")
                else:
                    print(f"    {t:10s}: {s_val:.4f} vs {b_val:.4f} "
                          f"({diff:+.4f})")

    # Convergence comparison
    print("\n" + "-" * 70)
    print("TRAINING EFFICIENCY")
    print("-" * 70)
    for cfg_name, res in all_results.items():
        t = res.get('train_time', 0)
        p = res.get('params', 0)
        in_acc = res.get('in_range', {}).get('overall', 0)
        print(f"  {cfg_name:20s}: {t:6.1f}s, {p:,} params, "
              f"in-range={in_acc:.4f}")


def save_results(all_results):
    save_data = {
        'phase': 'Phase 17: SPIRNOR-RoPE in Pretrained Language Models',
        'model': MODEL_NAME,
        'config': {
            'train_range': [2, 2000],
            'n_train_per_task': 8334,
            'n_eval_per_task': 1000,
            'epochs': 10,
            'batch_size': 64,
            'lr': 5e-5,
            'max_length': 48,
            'tasks': TASK_LIST,
            'spirnor_rope_primes': SPIRNOR_ROPE_PRIMES,
            'spirnor_value_primes': SPIRNOR_VALUE_PRIMES,
        },
        'eval_ranges': {
            'in_range': [2, 2000],
            'ood_2k_5k': [2001, 5000],
            'ood_5k_20k': [5001, 20000],
            'ood_20k_100k': [20001, 100000],
        },
        'results': {k: dict(v) if isinstance(v, OrderedDict) else v
                    for k, v in all_results.items()},
    }

    # Clean numpy types
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, OrderedDict):
            return {k: clean(v) for k, v in obj.items()}
        return obj

    save_data = clean(save_data)

    save_path = 'SPIRNOR_AI_PHASE17_RESULTS.json'
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {save_path}")


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 70)
    print("PHASE 17: SPIRNOR-RoPE IN PRETRAINED LANGUAGE MODELS")
    print(f"Model: {MODEL_NAME}")
    print("=" * 70)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Load base model to inspect architecture
    print("Loading base model for inspection...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    config = base_model.config
    print(f"  d_model: {config.hidden_size}")
    print(f"  n_heads: {config.num_attention_heads}")
    print(f"  n_layers: {config.num_hidden_layers}")
    print(f"  d_head: {config.hidden_size // config.num_attention_heads}")
    rotary_pct = getattr(config, 'rotary_pct',
                         getattr(config, 'rotary_percentage', 0.25))
    print(f"  rotary_pct: {rotary_pct}")
    rotary_ndims = base_model.gpt_neox.layers[0].attention.rotary_ndims
    n_freq_pairs = rotary_ndims // 2
    print(f"  Rotary dim: {rotary_ndims} -> {n_freq_pairs} frequency pairs")

    # Show standard frequencies (shared rotary_emb at model.gpt_neox level)
    inv_freq = base_model.gpt_neox.rotary_emb.inv_freq
    print(f"  Standard inv_freq: {inv_freq.tolist()}")
    del base_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ---- Generate Training Data ----
    print("\n" + "-" * 70)
    print("DATA GENERATION")
    print("-" * 70)

    train_range = (2, 2000)
    n_per_task = 8334  # ~50K total across 6 tasks

    print(f"Generating training data: {n_per_task} per task x {len(TASK_LIST)} "
          f"tasks = ~{n_per_task * len(TASK_LIST):,} examples")
    rng = np.random.RandomState(42)
    train_examples = generate_all_data(n_per_task, *train_range, rng)
    print(f"  Total training examples: {len(train_examples):,}")

    # Show samples
    for task_name in TASK_LIST:
        sample = [(t, p, a) for t, p, a in train_examples if t == task_name][:2]
        for t, p, a in sample:
            print(f"  [{t}] '{p}{a}'")

    # Tokenize training data
    print("\nTokenizing training data...")
    train_ids, train_mask, train_labels = tokenize_examples(
        train_examples, tokenizer, max_length=48)
    print(f"  Shape: input_ids={train_ids.shape}, "
          f"labels={train_labels.shape}")

    # ---- Eval Ranges ----
    eval_ranges = OrderedDict([
        ('in_range',     (2, 2000)),
        ('ood_2k_5k',    (2001, 5000)),
        ('ood_5k_20k',   (5001, 20000)),
        ('ood_20k_100k', (20001, 100000)),
    ])

    n_eval = 500  # Per task per range (6 tasks x 4 ranges x 500 = 12K evals)

    # ---- Run All Configs ----
    all_results = OrderedDict()

    for cfg_idx, (cfg_name, cfg) in enumerate(CONFIGS.items()):
        print(f"\n{'=' * 70}")
        print(f"[{cfg_idx + 1}/{len(CONFIGS)}] CONFIG: {cfg_name}")
        print(f"  {cfg['desc']}")
        print(f"{'=' * 70}")

        # Load fresh model
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

        # Apply SPIRNOR RoPE if needed
        if cfg['spirnor_rope']:
            replace_rope_with_spirnor(model)

        # Wrap with value augmentation if needed
        wrapped = PythiaWithSPIRNOR(model, tokenizer,
                                     use_value_aug=cfg['value_aug'])

        n_params = sum(p.numel() for p in wrapped.parameters())
        n_trainable = sum(p.numel() for p in wrapped.parameters()
                          if p.requires_grad)
        print(f"  Total params: {n_params:,}")
        print(f"  Trainable params: {n_trainable:,}")
        if cfg['value_aug']:
            aug_params = sum(p.numel() for p in wrapped.value_aug.parameters())
            print(f"  Value aug params: {aug_params:,}")

        # Move to device
        wrapped.to(device)

        # Fine-tune if needed
        train_time = 0.0
        if cfg['fine_tune']:
            print(f"\n  Fine-tuning ({10} epochs)...")
            train_time = train_model(wrapped, train_ids, train_mask,
                                      train_labels, epochs=10,
                                      batch_size=64, lr=2e-5)
            print(f"  Training time: {train_time:.1f}s")
        else:
            print("  (No fine-tuning — zero-shot evaluation)")

        # Quick debug: show 3 generation examples before full eval
        print("\n  Debug (3 examples):")
        for task_name in ['mod7', 'gcd_task', 'coprime']:
            gen_fn = TASKS[task_name]['gen']
            a, b = 466, 322
            prompt, answer = gen_fn(a, b)
            generated = wrapped.generate_answer(prompt, max_new_tokens=15)
            gen_clean = generated.split('\n')[0].split(' ')[0].strip()
            match = "OK" if gen_clean == answer.strip() else "MISS"
            print(f"    [{task_name}] prompt='{prompt.rstrip()}' "
                  f"expected='{answer}' got='{generated}' "
                  f"clean='{gen_clean}' [{match}]")

        # Evaluate
        print(f"\n  Evaluation ({n_eval} per task per range):")
        cfg_results = evaluate_all(wrapped, eval_ranges,
                                    n_eval_per_task=n_eval)
        cfg_results['params'] = n_params
        cfg_results['train_time'] = round(train_time, 1)
        cfg_results['trainable_params'] = n_trainable
        all_results[cfg_name] = cfg_results

        # Cleanup
        del wrapped, model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ---- Summary ----
    print_summary(all_results)
    save_results(all_results)

    return all_results


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    total_start = time.time()
    results = run_experiment()
    total_time = time.time() - total_start
    print(f"\nTotal experiment time: {total_time / 60:.1f} minutes")
    print("Phase 17 complete!")
