#!/usr/bin/env python3
"""
SPIRNOR AI Phase 17B + Phase 18: Multi-Seed Validation & Pythia-410M Scaling
Combined script for single Vast.ai RTX 4090 deployment.

Phase 17B: Multi-seed validation of Phase 17 findings (Pythia-70M)
  - 4 configs x 5 seeds = 20 training runs
  - Statistical hypothesis testing (paired t-test, Wilcoxon signed-rank)
  - Seeds: 42, 123, 456, 789, 2024 (matching Phase 7B convention)

Phase 18: Scale to Pythia-410M (single seed)
  - 5 configs x 1 seed on 405M parameter model
  - Same 6 tasks, same eval ranges
  - Tests whether SPIRNOR advantage holds at 5.8x model scale

Self-contained script for Vast.ai cloud deployment.
"""

import subprocess
import sys
import os

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
import argparse
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

SEEDS = [42, 123, 456, 789, 2024]

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
    gen_fn = TASKS[task_name]['gen']
    examples = []
    for _ in range(n_examples):
        a = int(rng.randint(lo, hi + 1))
        b = int(rng.randint(lo, hi + 1))
        prompt, answer = gen_fn(a, b)
        examples.append((prompt, answer))
    return examples

def generate_all_data(n_per_task, lo, hi, rng):
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
    n_float = float(max(n, 1))
    feats = [math.log(n_float)]
    for p in SPIRNOR_VALUE_PRIMES:
        angle = (TWO_PI / p) * n_float
        feats.append(math.sin(angle))
        feats.append(math.cos(angle))
    return torch.tensor(feats, dtype=torch.float32)

def extract_numbers_from_text(text):
    results = []
    for m in re.finditer(r'\d+', text):
        results.append((m.start(), m.end(), int(m.group())))
    return results

class SPIRNORValueAugmentation(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(N_VALUE_FEATURES, d_model, bias=False)
        nn.init.normal_(self.proj.weight, std=0.01)

    def forward(self, hidden_states, spirnor_mask, spirnor_features):
        aug = self.proj(spirnor_features)
        aug = aug * spirnor_mask.unsqueeze(-1).float()
        return hidden_states + aug

# ============================================================
# MODEL WRAPPER
# ============================================================

class PythiaWithSPIRNOR(nn.Module):
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
        B, S = input_ids.shape
        spirnor_mask = torch.zeros(B, S, dtype=torch.bool, device=input_ids.device)
        spirnor_features = torch.zeros(B, S, N_VALUE_FEATURES,
                                        device=input_ids.device)
        for b in range(B):
            tokens = input_ids[b]
            text = self.tokenizer.decode(tokens, skip_special_tokens=False)
            token_texts = [self.tokenizer.decode([t]) for t in tokens]
            char_pos = 0
            token_char_ranges = []
            for t_idx, t_text in enumerate(token_texts):
                start = char_pos
                char_pos += len(t_text)
                token_char_ranges.append((start, char_pos))
            eq_pos = text.find('=')
            if eq_pos == -1:
                eq_pos = len(text)
            numbers = extract_numbers_from_text(text[:eq_pos])
            for num_start, num_end, num_val in numbers:
                for t_idx, (t_start, t_end) in enumerate(token_char_ranges):
                    if t_idx >= S:
                        break
                    if t_start < num_end and t_end > num_start:
                        spirnor_mask[b, t_idx] = True
                        feats = compute_spirnor_features(num_val)
                        spirnor_features[b, t_idx] = feats.to(input_ids.device)
        return spirnor_mask, spirnor_features

    def forward(self, input_ids, attention_mask=None, labels=None):
        if self.use_value_aug and self.value_aug is not None:
            inputs_embeds = self.model.gpt_neox.embed_in(input_ids)
            spirnor_mask, spirnor_features = self.prepare_spirnor_inputs(input_ids)
            inputs_embeds = self.value_aug(inputs_embeds, spirnor_mask,
                                           spirnor_features)
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
        self.eval()
        # CRITICAL: Strip trailing space to match BPE tokenization during training.
        prompt_text = prompt_text.rstrip()
        input_ids = self.tokenizer.encode(prompt_text, return_tensors='pt')
        input_ids = input_ids.to(device)
        generated = input_ids.clone()
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
            token_text = self.tokenizer.decode(next_token[0])
            if '\n' in token_text or next_token.item() == self.tokenizer.eos_token_id:
                break
        gen_ids = generated[0, input_ids.shape[1]:]
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return gen_text.strip()

# ============================================================
# ROPE REPLACEMENT
# ============================================================

def replace_rope_with_spirnor(model):
    rotary_emb = model.gpt_neox.rotary_emb
    n_freqs = rotary_emb.inv_freq.shape[0]
    print(f"  RoPE replacement: {n_freqs} frequency pairs")
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
    print(f"  Primes used: {primes}")
    print(f"  CRT product: {math.prod(primes):,}")
    rotary_emb.inv_freq = spirnor_inv_freq
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
        prompt_stripped = prompt.rstrip()
        prompt_ids = tokenizer.encode(prompt_stripped, add_special_tokens=False)
        prompt_len = len(prompt_ids)
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100
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
    wrapped_model.to(device)
    wrapped_model.train()
    trainable = [p for p in wrapped_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01,
                                  eps=1e-6)
    n = len(input_ids)
    steps_per_epoch = (n + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch  # 1 epoch warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
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
            # VRAM check on first batch
            if global_step == 1 and device.type == 'cuda':
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"    VRAM: {allocated:.1f}GB alloc, {reserved:.1f}GB reserved, "
                      f"{total_mem:.1f}GB total ({reserved/total_mem*100:.0f}%)")
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

def evaluate_config(wrapped_model, task_name, lo, hi, n_eval=500, seed=42):
    wrapped_model.eval()
    rng = np.random.RandomState(seed)
    gen_fn = TASKS[task_name]['gen']
    correct = 0
    for _ in range(n_eval):
        a = int(rng.randint(lo, hi + 1))
        b = int(rng.randint(lo, hi + 1))
        prompt, answer = gen_fn(a, b)
        generated = wrapped_model.generate_answer(prompt, max_new_tokens=15)
        gen_clean = generated.split('\n')[0].split(' ')[0].strip()
        answer_clean = answer.strip()
        if gen_clean == answer_clean:
            correct += 1
    return correct / n_eval

def evaluate_all(wrapped_model, eval_ranges, n_eval_per_task=500):
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
        task_str = ' '.join(f"{t}={range_results[t]:.3f}" for t in TASK_LIST)
        print(f"    {range_name:14s}: overall={range_results['overall']:.4f}  "
              f"[{task_str}]")
    return results

# ============================================================
# EXPERIMENT CONFIGS
# ============================================================

EVAL_RANGES = OrderedDict([
    ('in_range',     (2, 2000)),
    ('ood_2k_5k',    (2001, 5000)),
    ('ood_5k_20k',   (5001, 20000)),
    ('ood_20k_100k', (20001, 100000)),
])

CONFIGS_FT = OrderedDict([
    ('baseline_ft', {
        'desc': 'Standard RoPE, fine-tuned',
        'spirnor_rope': False,
        'value_aug': False,
    }),
    ('spirnor_rope_ft', {
        'desc': 'SPIRNOR RoPE frequencies, fine-tuned',
        'spirnor_rope': True,
        'value_aug': False,
    }),
    ('spirnor_value_ft', {
        'desc': 'Standard RoPE + SPIRNOR value augmentation, fine-tuned',
        'spirnor_rope': False,
        'value_aug': True,
    }),
    ('spirnor_full_ft', {
        'desc': 'SPIRNOR RoPE + value augmentation, fine-tuned',
        'spirnor_rope': True,
        'value_aug': True,
    }),
])

# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

def compute_stats(arr):
    arr = np.array(arr, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    ci_95 = 2.776 * std / math.sqrt(n) if n > 1 else 0.0  # t-value for n=5, two-tailed 95%
    return {
        'mean': mean,
        'std': std,
        'ci_95': ci_95,
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'n': n,
        'values': arr.tolist(),
    }

def paired_t_test(vals_a, vals_b):
    """Paired t-test (same seed matched). More powerful than Welch's for n=5."""
    diffs = np.array(vals_a) - np.array(vals_b)
    n = len(diffs)
    mean_diff = np.mean(diffs)
    if n < 2:
        return 0.0, 1.0
    se_diff = np.std(diffs, ddof=1) / np.sqrt(n)
    if se_diff < 1e-10:
        return float('inf') if mean_diff > 0 else float('-inf'), 0.0
    t_stat = mean_diff / se_diff
    # p-value approximation using erfc (two-tailed)
    from math import erfc
    p_value = erfc(abs(t_stat) / math.sqrt(2))
    return float(t_stat), float(p_value)

def wilcoxon_signed_rank(vals_a, vals_b):
    """Wilcoxon signed-rank test (non-parametric). Returns W statistic and n_nonzero."""
    diffs = np.array(vals_a) - np.array(vals_b)
    nonzero = diffs[diffs != 0]
    n = len(nonzero)
    if n == 0:
        return 0, 0
    abs_diffs = np.abs(nonzero)
    ranks = np.argsort(np.argsort(abs_diffs)) + 1.0
    W_plus = float(np.sum(ranks[nonzero > 0]))
    W_minus = float(np.sum(ranks[nonzero < 0]))
    W = min(W_plus, W_minus)
    # For n=5: W=0 -> p=0.0625 (two-sided), W<=0 -> p=0.0312 (one-sided)
    return float(W), int(n)

# ============================================================
# CHECKPOINT UTILITIES
# ============================================================

def clean_for_json(obj):
    if isinstance(obj, (OrderedDict, dict)):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    return obj

def save_checkpoint(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(clean_for_json(data), f, indent=2)
    print(f"  Checkpoint saved: {filepath}")

def load_checkpoint(filepath):
    if os.path.exists(filepath):
        with open(filepath) as f:
            return json.load(f)
    return None


# ============================================================
# PHASE 17B: MULTI-SEED VALIDATION (PYTHIA-70M)
# ============================================================

def run_phase_17b(resume=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL_NAME = "EleutherAI/pythia-70m"
    CHECKPOINT_FILE = 'SPIRNOR_AI_PHASE17B_CHECKPOINT.json'
    RESULTS_FILE = 'SPIRNOR_AI_PHASE17B_RESULTS.json'
    N_PER_TASK = 8334
    TRAIN_RANGE = (2, 2000)
    BATCH_SIZE = 64
    LR = 2e-5
    EPOCHS = 10
    N_EVAL = 500

    print("\n" + "=" * 70)
    print("PHASE 17B: MULTI-SEED VALIDATION (PYTHIA-70M)")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Configs: {list(CONFIGS_FT.keys())}")
    print(f"  Total runs: {len(CONFIGS_FT)} x {len(SEEDS)} = {len(CONFIGS_FT) * len(SEEDS)}")
    print("=" * 70)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check for checkpoint
    completed_runs = {}
    if resume:
        ckpt = load_checkpoint(CHECKPOINT_FILE)
        if ckpt:
            completed_runs = ckpt.get('completed', {})
            print(f"  Resuming: {len(completed_runs)} runs already completed")

    # Structure: all_results[config_name] = [run_for_seed_0, run_for_seed_1, ...]
    all_results = {cfg: [] for cfg in CONFIGS_FT.keys()}

    total_runs = len(CONFIGS_FT) * len(SEEDS)
    run_idx = 0

    for seed_idx, seed in enumerate(SEEDS):
        for cfg_name, cfg in CONFIGS_FT.items():
            run_idx += 1
            run_key = f"{cfg_name}_seed{seed}"

            if run_key in completed_runs:
                print(f"\n[{run_idx}/{total_runs}] SKIP (cached): {cfg_name} seed={seed}")
                all_results[cfg_name].append(completed_runs[run_key])
                continue

            print(f"\n{'='*70}")
            print(f"[{run_idx}/{total_runs}] CONFIG: {cfg_name}, SEED: {seed}")
            print(f"  {cfg['desc']}")
            print(f"{'='*70}")

            # Set random seed for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Generate training data with this seed
            rng = np.random.RandomState(seed)
            train_examples = generate_all_data(N_PER_TASK, *TRAIN_RANGE, rng)
            print(f"  Training examples: {len(train_examples):,}")

            # Tokenize
            train_ids, train_mask, train_labels = tokenize_examples(
                train_examples, tokenizer, max_length=48)

            # Load fresh model
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

            # Apply SPIRNOR RoPE if needed
            if cfg['spirnor_rope']:
                replace_rope_with_spirnor(model)

            # Wrap
            wrapped = PythiaWithSPIRNOR(model, tokenizer,
                                         use_value_aug=cfg['value_aug'])
            wrapped.to(device)

            n_params = sum(p.numel() for p in wrapped.parameters())
            print(f"  Params: {n_params:,}")

            # Train
            print(f"\n  Fine-tuning ({EPOCHS} epochs)...")
            train_time = train_model(wrapped, train_ids, train_mask,
                                      train_labels, epochs=EPOCHS,
                                      batch_size=BATCH_SIZE, lr=LR)
            print(f"  Training time: {train_time:.1f}s")

            # Debug
            print("\n  Debug (3 examples):")
            for task_name in ['mod7', 'gcd_task', 'coprime']:
                gen_fn = TASKS[task_name]['gen']
                a, b = 466, 322
                prompt, answer = gen_fn(a, b)
                generated = wrapped.generate_answer(prompt, max_new_tokens=15)
                gen_clean = generated.split('\n')[0].split(' ')[0].strip()
                match = "OK" if gen_clean == answer.strip() else "MISS"
                print(f"    [{task_name}] expected='{answer}' got='{gen_clean}' [{match}]")

            # Evaluate
            print(f"\n  Evaluation ({N_EVAL} per task per range):")
            cfg_results = evaluate_all(wrapped, EVAL_RANGES,
                                        n_eval_per_task=N_EVAL)
            cfg_results['params'] = n_params
            cfg_results['train_time'] = round(train_time, 1)
            cfg_results['seed'] = seed

            all_results[cfg_name].append(cfg_results)
            completed_runs[run_key] = cfg_results

            # Save checkpoint
            save_checkpoint({
                'completed': completed_runs,
                'phase': 'Phase 17B',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            }, CHECKPOINT_FILE)

            # Cleanup
            del wrapped, model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    # ---- Statistical Analysis ----
    print_17b_summary(all_results)
    save_17b_results(all_results, RESULTS_FILE)
    return all_results


def print_17b_summary(all_results):
    range_keys = list(EVAL_RANGES.keys())
    ood_keys = range_keys[1:]

    print("\n" + "=" * 70)
    print("PHASE 17B: MULTI-SEED STATISTICAL SUMMARY")
    print("=" * 70)

    # Overall accuracy: mean +/- std
    print("\n" + "-" * 70)
    print("OVERALL ACCURACY: mean +/- std (5 seeds)")
    print("-" * 70)

    config_stats = OrderedDict()
    for config_name, seed_runs in all_results.items():
        if not seed_runs:
            continue
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

    # Average OOD per seed
    print("\n" + "-" * 70)
    print("AVERAGE OOD ACCURACY (mean of 3 OOD ranges, per seed)")
    print("-" * 70)

    avg_ood_per_config = OrderedDict()
    for config_name, seed_runs in all_results.items():
        if not seed_runs:
            continue
        per_seed = []
        for run in seed_runs:
            ood_vals = [run[k]['overall'] for k in ood_keys]
            per_seed.append(float(np.mean(ood_vals)))
        avg_ood_per_config[config_name] = per_seed
        stats = compute_stats(per_seed)
        seeds_str = ', '.join(f"{v:.3f}" for v in per_seed)
        print(f"  {config_name:20s}: {stats['mean']:.4f} +/- {stats['std']:.4f}  "
              f"[{seeds_str}]")

    # Statistical hypothesis tests
    print("\n" + "-" * 70)
    print("HYPOTHESIS TESTS (Paired t-test + Wilcoxon signed-rank)")
    print("-" * 70)

    hypotheses = [
        ('H1', 'spirnor_value_ft', 'baseline_ft',
         'value_ft > baseline_ft (main claim)'),
        ('H2', 'spirnor_value_ft', 'spirnor_full_ft',
         'value_ft ≈ full_ft (no compound benefit)'),
        ('H3', 'spirnor_rope_ft', 'baseline_ft',
         'rope_ft > baseline_ft (RoPE alone helps)'),
        ('H4', 'spirnor_value_ft', 'spirnor_rope_ft',
         'value_ft > rope_ft (value is primary driver)'),
    ]

    for h_id, name_a, name_b, desc in hypotheses:
        if name_a not in avg_ood_per_config or name_b not in avg_ood_per_config:
            continue
        vals_a = avg_ood_per_config[name_a]
        vals_b = avg_ood_per_config[name_b]
        t_stat, p_val = paired_t_test(vals_a, vals_b)
        W, n_nz = wilcoxon_signed_rank(vals_a, vals_b)
        mean_a = np.mean(vals_a)
        mean_b = np.mean(vals_b)
        diff = mean_a - mean_b
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"\n  {h_id}: {desc}")
        print(f"    {name_a}: {mean_a:.4f} vs {name_b}: {mean_b:.4f}  "
              f"diff={diff:+.4f}")
        print(f"    Paired t: t={t_stat:+.3f}, p={p_val:.6f} {sig}")
        print(f"    Wilcoxon: W={W:.0f}, n_nonzero={n_nz}")
        if h_id == 'H2':
            verdict = "CONFIRMED (no sig diff)" if p_val > 0.05 else "REJECTED (sig diff)"
        else:
            verdict = "CONFIRMED" if diff > 0 and p_val < 0.05 else \
                      "TREND" if diff > 0 else "REJECTED"
        print(f"    -> {verdict}")

    # Per-task at 2K-5K
    print("\n" + "-" * 70)
    print("PER-TASK ACCURACY AT 2K-5K: mean +/- std")
    print("-" * 70)

    for task in TASK_LIST:
        print(f"\n  {task}:")
        for config_name, seed_runs in all_results.items():
            if not seed_runs:
                continue
            vals = [run['ood_2k_5k'][task] for run in seed_runs]
            stats = compute_stats(vals)
            print(f"    {config_name:20s}: {stats['mean']:.3f} +/- {stats['std']:.3f}  "
                  f"[{stats['min']:.3f} - {stats['max']:.3f}]")


def save_17b_results(all_results, filepath):
    ood_keys = ['ood_2k_5k', 'ood_5k_20k', 'ood_20k_100k']
    avg_ood = {}
    for config_name, seed_runs in all_results.items():
        if not seed_runs:
            continue
        per_seed = []
        for run in seed_runs:
            ood_vals = [run[k]['overall'] for k in ood_keys]
            per_seed.append(float(np.mean(ood_vals)))
        avg_ood[config_name] = compute_stats(per_seed)

    save_data = {
        'phase': 'Phase 17B: Multi-Seed Statistical Validation',
        'model': 'EleutherAI/pythia-70m',
        'seeds': SEEDS,
        'config': {
            'train_range': [2, 2000],
            'n_train_per_task': 8334,
            'n_eval_per_task': 500,
            'epochs': 10,
            'batch_size': 64,
            'lr': 2e-5,
            'max_length': 48,
            'tasks': TASK_LIST,
        },
        'per_seed_results': all_results,
        'avg_ood_stats': avg_ood,
    }
    save_checkpoint(save_data, filepath)
    print(f"\nPhase 17B results saved to: {filepath}")


# ============================================================
# PHASE 18: PYTHIA-410M SCALING (SINGLE SEED)
# ============================================================

def run_phase_18():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL_NAME = "EleutherAI/pythia-410m"
    RESULTS_FILE = 'SPIRNOR_AI_PHASE18_RESULTS.json'
    N_PER_TASK = 8334
    TRAIN_RANGE = (2, 2000)
    BATCH_SIZE = 32  # Reduced from 64 for memory
    LR = 1e-5  # Reduced from 2e-5 for larger model stability
    EPOCHS = 10
    N_EVAL = 500
    SEED = 42

    CONFIGS_18 = OrderedDict([
        ('zero_shot', {
            'desc': 'Pretrained Pythia-410M (no fine-tuning)',
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

    print("\n" + "=" * 70)
    print("PHASE 18: PYTHIA-410M SCALING")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Seed: {SEED}")
    print(f"  Batch size: {BATCH_SIZE}, LR: {LR}")
    print("=" * 70)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Inspect architecture
    print("\nLoading model for architecture inspection...")
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
    inv_freq = base_model.gpt_neox.rotary_emb.inv_freq
    print(f"  Standard inv_freq: {inv_freq.tolist()}")
    n_total_params = sum(p.numel() for p in base_model.parameters())
    print(f"  Total params: {n_total_params:,}")
    del base_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Generate training data
    print("\nGenerating training data...")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.RandomState(SEED)
    train_examples = generate_all_data(N_PER_TASK, *TRAIN_RANGE, rng)
    print(f"  Training examples: {len(train_examples):,}")

    # Tokenize
    print("Tokenizing...")
    train_ids, train_mask, train_labels = tokenize_examples(
        train_examples, tokenizer, max_length=48)

    # Run all configs
    all_results = OrderedDict()

    for cfg_idx, (cfg_name, cfg) in enumerate(CONFIGS_18.items()):
        print(f"\n{'='*70}")
        print(f"[{cfg_idx+1}/{len(CONFIGS_18)}] CONFIG: {cfg_name}")
        print(f"  {cfg['desc']}")
        print(f"{'='*70}")

        # Load fresh model
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

        if cfg['spirnor_rope']:
            replace_rope_with_spirnor(model)

        wrapped = PythiaWithSPIRNOR(model, tokenizer,
                                     use_value_aug=cfg['value_aug'])

        n_params = sum(p.numel() for p in wrapped.parameters())
        n_trainable = sum(p.numel() for p in wrapped.parameters()
                          if p.requires_grad)
        print(f"  Total params: {n_params:,}")
        if cfg['value_aug']:
            aug_params = sum(p.numel() for p in wrapped.value_aug.parameters())
            print(f"  Value aug params: {aug_params:,}")

        wrapped.to(device)

        train_time = 0.0
        if cfg['fine_tune']:
            print(f"\n  Fine-tuning ({EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR})...")
            train_time = train_model(wrapped, train_ids, train_mask,
                                      train_labels, epochs=EPOCHS,
                                      batch_size=BATCH_SIZE, lr=LR)
            print(f"  Training time: {train_time:.1f}s")
        else:
            print("  (No fine-tuning — zero-shot evaluation)")

        # Debug
        print("\n  Debug (3 examples):")
        for task_name in ['mod7', 'gcd_task', 'coprime']:
            gen_fn = TASKS[task_name]['gen']
            a, b = 466, 322
            prompt, answer = gen_fn(a, b)
            generated = wrapped.generate_answer(prompt, max_new_tokens=15)
            gen_clean = generated.split('\n')[0].split(' ')[0].strip()
            match = "OK" if gen_clean == answer.strip() else "MISS"
            print(f"    [{task_name}] expected='{answer}' got='{gen_clean}' [{match}]")

        # Evaluate
        print(f"\n  Evaluation ({N_EVAL} per task per range):")
        cfg_results = evaluate_all(wrapped, EVAL_RANGES,
                                    n_eval_per_task=N_EVAL)
        cfg_results['params'] = n_params
        cfg_results['train_time'] = round(train_time, 1)
        cfg_results['trainable_params'] = n_trainable
        all_results[cfg_name] = cfg_results

        del wrapped, model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Summary and save
    print_18_summary(all_results)
    save_18_results(all_results, RESULTS_FILE)
    return all_results


def print_18_summary(all_results):
    range_keys = list(EVAL_RANGES.keys())
    range_labels = ['In-Range', '2K-5K', '5K-20K', '20K-100K']

    print("\n" + "=" * 70)
    print("PHASE 18 RESULTS SUMMARY (PYTHIA-410M)")
    print("=" * 70)

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

    for cfg_name, res in all_results.items():
        line = f"  {cfg_name:20s}"
        for t in TASK_LIST:
            val = res.get('ood_2k_5k', {}).get(t, 0)
            line += f" | {val:.4f}    "
        print(line)

    # Compare to Phase 17 (70M)
    if 'baseline_ft' in all_results and 'spirnor_value_ft' in all_results:
        print("\n" + "-" * 70)
        print("SPIRNOR ADVANTAGE (value_ft vs baseline_ft, 2K-5K)")
        print("-" * 70)
        for t in TASK_LIST:
            s_val = all_results['spirnor_value_ft'].get('ood_2k_5k', {}).get(t, 0)
            b_val = all_results['baseline_ft'].get('ood_2k_5k', {}).get(t, 0)
            diff = s_val - b_val
            ratio = s_val / b_val if b_val > 0 else float('inf')
            print(f"    {t:10s}: {s_val:.4f} vs {b_val:.4f} "
                  f"({diff:+.4f}, {ratio:.1f}x)")


def save_18_results(all_results, filepath):
    save_data = {
        'phase': 'Phase 18: SPIRNOR in Pythia-410M',
        'model': 'EleutherAI/pythia-410m',
        'config': {
            'train_range': [2, 2000],
            'n_train_per_task': 8334,
            'n_eval_per_task': 500,
            'epochs': 10,
            'batch_size': 32,
            'lr': 1e-5,
            'max_length': 48,
            'tasks': TASK_LIST,
            'spirnor_rope_primes': SPIRNOR_ROPE_PRIMES,
            'spirnor_value_primes': SPIRNOR_VALUE_PRIMES,
        },
        'eval_ranges': {k: list(v) for k, v in EVAL_RANGES.items()},
        'results': all_results,
    }
    save_checkpoint(save_data, filepath)
    print(f"\nPhase 18 results saved to: {filepath}")


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SPIRNOR AI Phase 17B + Phase 18')
    parser.add_argument('--phase', choices=['17b', '18', 'both'],
                        default='both',
                        help='Which phase to run (default: both)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint (Phase 17B only)')
    args = parser.parse_args()

    total_start = time.time()

    if args.phase in ['17b', 'both']:
        print("\n" + "#" * 70)
        print("# STARTING PHASE 17B: MULTI-SEED VALIDATION")
        print("#" * 70)
        p17b_start = time.time()
        run_phase_17b(resume=args.resume)
        p17b_time = time.time() - p17b_start
        print(f"\nPhase 17B total time: {p17b_time / 60:.1f} minutes")

    if args.phase in ['18', 'both']:
        print("\n" + "#" * 70)
        print("# STARTING PHASE 18: PYTHIA-410M SCALING")
        print("#" * 70)
        p18_start = time.time()
        run_phase_18()
        p18_time = time.time() - p18_start
        print(f"\nPhase 18 total time: {p18_time / 60:.1f} minutes")

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"TOTAL EXPERIMENT TIME: {total_time / 60:.1f} minutes "
          f"({total_time / 3600:.1f} hours)")
    print(f"{'='*70}")
    print("All experiments complete!")
