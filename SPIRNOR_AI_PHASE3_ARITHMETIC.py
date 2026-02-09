#!/usr/bin/env python3
"""
SPIRNOR AI Phase 3: Numeric Embedding for Arithmetic Reasoning
===============================================================

Tests whether SPIRNOR embeddings give a transformer a structural
advantage on tasks where multiplicative/modular structure matters.

Core insight: SPIRNOR maps integers via r=ln(n), theta=(C*n) mod 2pi.
  - ln(a*b) = ln(a) + ln(b)  -> multiplication is ADDITIVE in radial component
  - Numbers sharing factors align angularly at certain C values
  - This structure is exactly what arithmetic tasks require

Tasks:
  1. Modular multiplication: (a, b) -> (a*b) mod p
  2. GCD classification:     (a, b) -> gcd(a,b) bucketed
  3. Smallest prime factor:   n     -> spf(n) bucketed
  4. Semiprime detection:     n     -> is_semiprime(n)

Embedding methods compared (all at matched dimensionality):
  1. Learned embedding     - standard nn.Embedding, no structure
  2. SPIRNOR embedding     - fixed SPIRNOR mapping, structure baked in
  3. SPIRNOR + learned     - SPIRNOR structure + learnable residual
  4. Sinusoidal embedding  - sin/cos at geometric frequencies (like PE)

Each feeds into the same small transformer. The question:
does SPIRNOR's number-theoretic structure help?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
import time
import json
import os
from typing import List, Tuple, Dict
from dataclasses import dataclass

# ===== SPIRNOR CONSTANTS =====
PHI = (1 + math.sqrt(5)) / 2

# Use the top constants from Phase 1 + key ones from Phase 2
WINDING_CONSTANTS = [
    math.pi,                    # dominant in Phase 1
    math.sqrt(2),               # 2nd in Phase 1, top attention in Phase 2
    PHI ** 2,                   # strong standalone, top attention
    math.e,                     # strong attention weight
    2 * math.pi / (PHI ** 2),   # golden angle
    PHI,                        # golden ratio
    math.log(2),                # ln2
    math.pi / math.e,           # pi/e
]

CONSTANT_NAMES = [
    'pi', 'sqrt2', 'phi_sq', 'e', 'golden_angle', 'phi', 'ln2', 'pi_e'
]


# ===== PRIME UTILITIES =====

def sieve(limit: int) -> np.ndarray:
    """Boolean mask of primes up to limit."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0:2] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return is_prime


def smallest_prime_factor_array(limit: int) -> np.ndarray:
    """Compute smallest prime factor for every number up to limit."""
    spf = np.arange(limit + 1)  # spf[i] = i initially
    for i in range(2, int(np.sqrt(limit)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf


def count_prime_factors_array(limit: int) -> np.ndarray:
    """Count distinct prime factors for every number up to limit."""
    omega = np.zeros(limit + 1, dtype=np.int32)
    for p in range(2, limit + 1):
        if omega[p] == 0:  # p hasn't been divided yet, so it's prime
            # Actually need a proper sieve approach
            pass
    # Use spf-based approach
    spf = smallest_prime_factor_array(limit)
    for n in range(2, limit + 1):
        temp = n
        count = 0
        while temp > 1:
            p = spf[temp]
            count += 1
            while temp % p == 0:
                temp //= p
        omega[n] = count
    return omega


# ===== EMBEDDING METHODS =====

class LearnedNumberEmbedding(nn.Module):
    """Standard learned embedding. No structure, pure memorization."""

    def __init__(self, max_num: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(max_num + 1, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)

    def name(self): return "Learned"


class SPIRNORNumberEmbedding(nn.Module):
    """
    Fixed SPIRNOR embedding. Maps each integer through the SPIRNOR formula
    at K winding constants, producing a structured representation.

    For number n and constant C_k:
        r     = ln(n)
        theta = (C_k * n) mod 2pi
        phi   = (PHI * n) mod 2pi

        features = [x, y, z, sin(theta), cos(theta), sin(phi), cos(phi)]

    Key structural properties:
        - r is ADDITIVE under multiplication: ln(ab) = ln(a) + ln(b)
        - theta alignment reveals shared factors at certain C values
        - phi (golden ratio) provides quasi-uniform angular coverage
    """

    def __init__(self, max_num: int, d_model: int,
                 constants: List[float] = None):
        super().__init__()

        if constants is None:
            constants = WINDING_CONSTANTS
        self.constants = constants
        self.K = len(constants)

        # 7 features per constant
        features_per_c = 7
        raw_dim = self.K * features_per_c

        # Project to d_model if needed
        if raw_dim != d_model:
            self.proj = nn.Linear(raw_dim, d_model, bias=False)
        else:
            self.proj = nn.Identity()

        self.raw_dim = raw_dim
        self.d_model = d_model

        # Precompute embeddings for all numbers up to max_num
        embeddings = self._compute_all(max_num)
        self.register_buffer('embeddings', embeddings)

    def _compute_all(self, max_num: int) -> torch.Tensor:
        """Precompute SPIRNOR embeddings for 0..max_num."""
        n_vals = torch.arange(max_num + 1, dtype=torch.float32)

        all_feats = []
        for C in self.constants:
            # Avoid log(0)
            r = torch.log(torch.clamp(n_vals, min=1.0))
            theta = (C * n_vals) % (2 * math.pi)
            phi = (PHI * n_vals) % (2 * math.pi)

            x = r * torch.sin(theta) * torch.cos(phi)
            y = r * torch.sin(theta) * torch.sin(phi)
            z = r * torch.cos(theta)

            feats = torch.stack([
                x, y, z,
                torch.sin(theta), torch.cos(theta),
                torch.sin(phi), torch.cos(phi),
            ], dim=-1)  # (max_num+1, 7)

            all_feats.append(feats)

        # Concatenate: (max_num+1, K*7)
        return torch.cat(all_feats, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) of integer indices
        raw = self.embeddings[x]  # (batch, seq_len, raw_dim)
        return self.proj(raw)

    def name(self): return "SPIRNOR"


class SPIRNORPlusLearnedEmbedding(nn.Module):
    """
    SPIRNOR provides structural foundation, learned residual adds flexibility.
    embed(n) = project(spirnor(n)) + learned(n)
    """

    def __init__(self, max_num: int, d_model: int,
                 constants: List[float] = None):
        super().__init__()
        self.spirnor = SPIRNORNumberEmbedding(max_num, d_model, constants)
        self.learned = nn.Embedding(max_num + 1, d_model)
        # Initialize learned part small so SPIRNOR structure dominates initially
        nn.init.normal_(self.learned.weight, std=0.01)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spirnor(x) + self.learned(x)

    def name(self): return "SPIRNOR+Learned"


class SinusoidalNumberEmbedding(nn.Module):
    """
    Sinusoidal embedding applied to number VALUES (not positions).
    Same as transformer PE but treating the number as the "position".
    This is the structured baseline - similar idea, different structure.

    PE(n, 2i)   = sin(n / base^(2i/d))
    PE(n, 2i+1) = cos(n / base^(2i/d))
    """

    def __init__(self, max_num: int, d_model: int, base: float = 100.0):
        super().__init__()
        self.d_model = d_model

        # Precompute
        n_vals = torch.arange(max_num + 1, dtype=torch.float32).unsqueeze(1)  # (N, 1)
        dim_indices = torch.arange(d_model, dtype=torch.float32).unsqueeze(0)  # (1, d)

        # Frequencies: geometric progression
        freqs = 1.0 / (base ** (2 * (dim_indices // 2) / d_model))  # (1, d)
        angles = n_vals * freqs  # (N, d)

        # sin for even, cos for odd
        embeddings = torch.zeros(max_num + 1, d_model)
        embeddings[:, 0::2] = torch.sin(angles[:, 0::2])
        embeddings[:, 1::2] = torch.cos(angles[:, 1::2])

        self.register_buffer('embeddings', embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embeddings[x]

    def name(self): return "Sinusoidal"


# ===== TRANSFORMER MODEL =====

class ArithmeticTransformer(nn.Module):
    """
    Small transformer for arithmetic tasks.
    Takes embedded number sequence, processes with self-attention,
    pools, and predicts output class.
    """

    def __init__(self, embedding: nn.Module, d_model: int,
                 n_classes: int, n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.embedding = embedding
        self.d_model = d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len) integer tensor
        returns: (batch, n_classes) logits
        """
        emb = self.embedding(x)          # (batch, seq_len, d_model)
        out = self.transformer(emb)      # (batch, seq_len, d_model)
        pooled = out.mean(dim=1)         # (batch, d_model) - mean pool
        return self.head(pooled)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===== DATA GENERATION =====

def generate_modmul_data(n_samples: int, num_range: Tuple[int, int],
                         p: int = 7, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Modular multiplication: input (a, b), output (a*b) mod p.
    This directly tests whether the embedding captures multiplicative structure.
    """
    rng = np.random.RandomState(seed)
    lo, hi = num_range
    a = rng.randint(lo, hi + 1, size=n_samples)
    b = rng.randint(lo, hi + 1, size=n_samples)
    y = (a * b) % p

    X = torch.tensor(np.stack([a, b], axis=1), dtype=torch.long)
    Y = torch.tensor(y, dtype=torch.long)
    return X, Y


def generate_gcd_data(n_samples: int, num_range: Tuple[int, int],
                      seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GCD prediction: input (a, b), output gcd(a,b) bucketed.
    Buckets: {1, 2, 3, 4, 5, 6, 7-10, 11-20, 21+}
    """
    rng = np.random.RandomState(seed)
    lo, hi = num_range
    a = rng.randint(lo, hi + 1, size=n_samples)
    b = rng.randint(lo, hi + 1, size=n_samples)

    gcds = np.array([math.gcd(int(a[i]), int(b[i])) for i in range(n_samples)])

    # Bucket GCDs
    def bucket(g):
        if g <= 6: return g - 1      # 0-5 for gcd 1-6
        if g <= 10: return 6          # 6 for gcd 7-10
        if g <= 20: return 7          # 7 for gcd 11-20
        return 8                       # 8 for gcd 21+

    y = np.array([bucket(g) for g in gcds])

    X = torch.tensor(np.stack([a, b], axis=1), dtype=torch.long)
    Y = torch.tensor(y, dtype=torch.long)
    return X, Y


def generate_spf_data(n_samples: int, num_range: Tuple[int, int],
                      seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Smallest prime factor: input n, output spf(n) bucketed.
    Buckets: {2, 3, 5, 7, 11, 13, 17-29, 31+, is_prime}
    """
    lo, hi = num_range
    spf_arr = smallest_prime_factor_array(hi)

    rng = np.random.RandomState(seed)
    nums = rng.randint(lo, hi + 1, size=n_samples)

    def bucket_spf(n):
        s = spf_arr[n]
        if s == n:  return 8  # n is prime (spf = itself)
        if s == 2:  return 0
        if s == 3:  return 1
        if s == 5:  return 2
        if s == 7:  return 3
        if s == 11: return 4
        if s == 13: return 5
        if s <= 29: return 6
        return 7

    y = np.array([bucket_spf(int(n)) for n in nums])

    X = torch.tensor(nums, dtype=torch.long).unsqueeze(1)  # (N, 1)
    Y = torch.tensor(y, dtype=torch.long)
    return X, Y


def generate_semiprime_data(n_samples: int, num_range: Tuple[int, int],
                            seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Semiprime detection: input n, output 0/1.
    A semiprime has exactly 2 prime factors (with multiplicity).
    """
    lo, hi = num_range
    spf_arr = smallest_prime_factor_array(hi)

    rng = np.random.RandomState(seed)
    nums = rng.randint(lo, hi + 1, size=n_samples)

    def is_semiprime(n):
        if n < 4: return 0
        # Count total prime factors with multiplicity
        temp = int(n)
        count = 0
        while temp > 1:
            p = spf_arr[temp]
            temp //= p
            count += 1
        return 1 if count == 2 else 0

    y = np.array([is_semiprime(int(n)) for n in nums])

    X = torch.tensor(nums, dtype=torch.long).unsqueeze(1)
    Y = torch.tensor(y, dtype=torch.long)
    return X, Y


# ===== TRAINING =====

@dataclass
class TaskResult:
    task: str
    embedding: str
    n_params: int
    train_acc: float
    test_in_acc: float
    test_out_acc: float
    test_in_f1: float
    test_out_f1: float
    train_time: float
    epochs: int
    final_loss: float


def train_and_evaluate(
    task_name: str,
    embedding_class,
    X_train: torch.Tensor, y_train: torch.Tensor,
    X_test_in: torch.Tensor, y_test_in: torch.Tensor,
    X_test_out: torch.Tensor, y_test_out: torch.Tensor,
    max_num: int, d_model: int, n_classes: int,
    epochs: int = 30, batch_size: int = 256, lr: float = 0.001,
    **embed_kwargs
) -> TaskResult:
    """Train a model and evaluate on in-range and out-of-range test sets."""

    # Create embedding
    embed = embedding_class(max_num, d_model, **embed_kwargs)
    embed_name = embed.name()

    # Create model
    model = ArithmeticTransformer(
        embedding=embed, d_model=d_model, n_classes=n_classes,
        n_heads=4, n_layers=2, d_ff=128, dropout=0.1
    )
    n_params = model.count_params()

    print(f"    {embed_name:20s} | {n_params:>8,} params | ", end="", flush=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(TensorDataset(X_train, y_train),
                        batch_size=batch_size, shuffle=True)

    t0 = time.time()
    final_loss = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_b, y_b in loader:
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        final_loss = epoch_loss / n_batches

    train_time = time.time() - t0

    # Evaluate
    model.eval()
    from sklearn.metrics import f1_score as sklearn_f1, accuracy_score

    with torch.no_grad():
        # Train accuracy
        train_pred = model(X_train).argmax(dim=-1).numpy()
        train_acc = accuracy_score(y_train.numpy(), train_pred)

        # Test in-range
        test_in_pred = model(X_test_in).argmax(dim=-1).numpy()
        test_in_acc = accuracy_score(y_test_in.numpy(), test_in_pred)
        avg = 'binary' if n_classes == 2 else 'weighted'
        test_in_f1 = sklearn_f1(y_test_in.numpy(), test_in_pred,
                                average=avg, zero_division=0)

        # Test out-of-range
        test_out_pred = model(X_test_out).argmax(dim=-1).numpy()
        test_out_acc = accuracy_score(y_test_out.numpy(), test_out_pred)
        test_out_f1 = sklearn_f1(y_test_out.numpy(), test_out_pred,
                                 average=avg, zero_division=0)

    print(f"in={test_in_acc:.4f} out={test_out_acc:.4f} "
          f"f1_in={test_in_f1:.4f} f1_out={test_out_f1:.4f} "
          f"({train_time:.1f}s)")

    return TaskResult(
        task=task_name, embedding=embed_name, n_params=n_params,
        train_acc=train_acc, test_in_acc=test_in_acc,
        test_out_acc=test_out_acc, test_in_f1=test_in_f1,
        test_out_f1=test_out_f1, train_time=train_time,
        epochs=epochs, final_loss=final_loss,
    )


# ===== MAIN EXPERIMENT =====

def run_task(task_name: str, gen_func, train_range: Tuple[int, int],
             test_in_range: Tuple[int, int], test_out_range: Tuple[int, int],
             n_classes: int, n_train: int = 20000, n_test: int = 5000,
             d_model: int = 56, epochs: int = 30,
             **gen_kwargs) -> List[TaskResult]:
    """Run all embedding methods on one task."""

    print(f"\n{'='*70}")
    print(f"TASK: {task_name}")
    print(f"{'='*70}")
    print(f"  Train range:    {train_range}")
    print(f"  Test in-range:  {test_in_range}")
    print(f"  Test out-range: {test_out_range}")
    print(f"  Classes: {n_classes}, d_model: {d_model}, epochs: {epochs}")

    # Generate data
    X_train, y_train = gen_func(n_train, train_range, seed=42, **gen_kwargs)
    X_test_in, y_test_in = gen_func(n_test, test_in_range, seed=123, **gen_kwargs)
    X_test_out, y_test_out = gen_func(n_test, test_out_range, seed=456, **gen_kwargs)

    max_num = max(test_out_range[1], train_range[1]) + 1

    print(f"  Train: {X_train.shape}, Test-in: {X_test_in.shape}, "
          f"Test-out: {X_test_out.shape}")
    print(f"  Train label dist: {np.bincount(y_train.numpy(), minlength=n_classes)}")
    print()
    print(f"    {'Embedding':20s} | {'Params':>8s} | Results")
    print(f"    {'-'*60}")

    results = []

    # 1. Learned embedding
    r = train_and_evaluate(
        task_name, LearnedNumberEmbedding,
        X_train, y_train, X_test_in, y_test_in, X_test_out, y_test_out,
        max_num, d_model, n_classes, epochs=epochs
    )
    results.append(r)

    # 2. SPIRNOR embedding (fixed)
    r = train_and_evaluate(
        task_name, SPIRNORNumberEmbedding,
        X_train, y_train, X_test_in, y_test_in, X_test_out, y_test_out,
        max_num, d_model, n_classes, epochs=epochs,
        constants=WINDING_CONSTANTS
    )
    results.append(r)

    # 3. SPIRNOR + Learned
    r = train_and_evaluate(
        task_name, SPIRNORPlusLearnedEmbedding,
        X_train, y_train, X_test_in, y_test_in, X_test_out, y_test_out,
        max_num, d_model, n_classes, epochs=epochs,
        constants=WINDING_CONSTANTS
    )
    results.append(r)

    # 4. Sinusoidal embedding
    r = train_and_evaluate(
        task_name, SinusoidalNumberEmbedding,
        X_train, y_train, X_test_in, y_test_in, X_test_out, y_test_out,
        max_num, d_model, n_classes, epochs=epochs
    )
    results.append(r)

    return results


def run_all():
    """Run the complete Phase 3 experiment."""
    print("=" * 70)
    print("SPIRNOR AI PHASE 3: NUMERIC EMBEDDING FOR ARITHMETIC REASONING")
    print("=" * 70)
    print()
    print("Testing whether SPIRNOR embeddings give transformers a structural")
    print("advantage on tasks where multiplicative/modular structure matters.")
    print()
    print(f"SPIRNOR constants used ({len(WINDING_CONSTANTS)}):")
    for name, val in zip(CONSTANT_NAMES, WINDING_CONSTANTS):
        print(f"  {name:14s} = {val:.6f}")

    all_results = []

    # --- Task 1: Modular Multiplication ---
    # (a*b) mod 7 - directly tests multiplicative structure
    # SPIRNOR advantage: ln(a*b) = ln(a) + ln(b), angular alignment of multiples
    results = run_task(
        "ModMul (mod 7)",
        generate_modmul_data,
        train_range=(2, 500),
        test_in_range=(2, 500),
        test_out_range=(501, 1000),
        n_classes=7, n_train=20000, n_test=5000,
        d_model=56, epochs=30, p=7,
    )
    all_results.extend(results)

    # --- Task 1b: Modular Multiplication mod 13 (harder) ---
    results = run_task(
        "ModMul (mod 13)",
        generate_modmul_data,
        train_range=(2, 500),
        test_in_range=(2, 500),
        test_out_range=(501, 1000),
        n_classes=13, n_train=20000, n_test=5000,
        d_model=56, epochs=30, p=13,
    )
    all_results.extend(results)

    # --- Task 2: GCD Classification ---
    # Directly tests factor-sharing detection
    # SPIRNOR advantage: numbers sharing factors align at specific C values
    results = run_task(
        "GCD Classification",
        generate_gcd_data,
        train_range=(2, 500),
        test_in_range=(2, 500),
        test_out_range=(501, 1000),
        n_classes=9, n_train=20000, n_test=5000,
        d_model=56, epochs=30,
    )
    all_results.extend(results)

    # --- Task 3: Smallest Prime Factor ---
    # Tests prime structure detection
    results = run_task(
        "Smallest Prime Factor",
        generate_spf_data,
        train_range=(2, 500),
        test_in_range=(2, 500),
        test_out_range=(501, 2000),
        n_classes=9, n_train=20000, n_test=5000,
        d_model=56, epochs=30,
    )
    all_results.extend(results)

    # --- Task 4: Semiprime Detection ---
    results = run_task(
        "Semiprime Detection",
        generate_semiprime_data,
        train_range=(4, 500),
        test_in_range=(4, 500),
        test_out_range=(501, 2000),
        n_classes=2, n_train=20000, n_test=5000,
        d_model=56, epochs=30,
    )
    all_results.extend(results)

    # ===== SUMMARY TABLE =====
    print(f"\n\n{'='*70}")
    print("PHASE 3 RESULTS SUMMARY")
    print(f"{'='*70}")

    # Group by task
    tasks = {}
    for r in all_results:
        if r.task not in tasks:
            tasks[r.task] = []
        tasks[r.task].append(r)

    for task_name, task_results in tasks.items():
        print(f"\n  {task_name}:")
        print(f"    {'Embedding':20s} | {'Params':>8s} | {'In-Acc':>7s} | "
              f"{'Out-Acc':>7s} | {'In-F1':>7s} | {'Out-F1':>7s} | "
              f"{'Generalization':>14s}")
        print(f"    {'-'*90}")

        best_in = max(r.test_in_acc for r in task_results)
        best_out = max(r.test_out_acc for r in task_results)

        for r in task_results:
            gen_gap = r.test_in_acc - r.test_out_acc
            in_marker = " *" if r.test_in_acc == best_in else "  "
            out_marker = " *" if r.test_out_acc == best_out else "  "
            print(f"    {r.embedding:20s} | {r.n_params:>8,} | "
                  f"{r.test_in_acc:>6.4f}{in_marker}| "
                  f"{r.test_out_acc:>6.4f}{out_marker}| "
                  f"{r.test_in_f1:>7.4f} | {r.test_out_f1:>7.4f} | "
                  f"gap={gen_gap:+.4f}")

    # ===== SPIRNOR ADVANTAGE ANALYSIS =====
    print(f"\n\n{'='*70}")
    print("SPIRNOR ADVANTAGE ANALYSIS")
    print(f"{'='*70}")
    print("\nComparing SPIRNOR vs each baseline per task:")
    print("(positive = SPIRNOR wins, negative = baseline wins)\n")

    spirnor_wins = 0
    spirnor_total = 0

    for task_name, task_results in tasks.items():
        spirnor_r = next((r for r in task_results if r.embedding == "SPIRNOR"), None)
        if spirnor_r is None:
            continue

        print(f"  {task_name}:")
        for r in task_results:
            if r.embedding == "SPIRNOR":
                continue

            # Compare out-of-range accuracy (generalization is key)
            delta_out = spirnor_r.test_out_acc - r.test_out_acc
            delta_in = spirnor_r.test_in_acc - r.test_in_acc

            # Compare generalization gap
            spirnor_gap = spirnor_r.test_in_acc - spirnor_r.test_out_acc
            other_gap = r.test_in_acc - r.test_out_acc
            delta_gap = other_gap - spirnor_gap  # positive = SPIRNOR generalizes better

            winner_out = "SPIRNOR" if delta_out > 0.005 else (
                r.embedding if delta_out < -0.005 else "TIE")
            winner_gen = "SPIRNOR" if delta_gap > 0.005 else (
                r.embedding if delta_gap < -0.005 else "TIE")

            if delta_out > 0.005:
                spirnor_wins += 1
            spirnor_total += 1

            print(f"    vs {r.embedding:20s}: "
                  f"out-acc {delta_out:+.4f} ({winner_out}), "
                  f"gen-gap {delta_gap:+.4f} ({winner_gen})")

    if spirnor_total > 0:
        win_rate = spirnor_wins / spirnor_total
        print(f"\n  SPIRNOR out-of-range win rate: {spirnor_wins}/{spirnor_total} "
              f"= {win_rate:.0%}")

    # ===== HYBRID ANALYSIS =====
    print(f"\n{'='*70}")
    print("HYBRID (SPIRNOR+Learned) ANALYSIS")
    print(f"{'='*70}")
    print("Does adding SPIRNOR structure to learned embeddings help?\n")

    for task_name, task_results in tasks.items():
        learned_r = next((r for r in task_results if r.embedding == "Learned"), None)
        hybrid_r = next((r for r in task_results if r.embedding == "SPIRNOR+Learned"), None)
        if learned_r and hybrid_r:
            delta_in = hybrid_r.test_in_acc - learned_r.test_in_acc
            delta_out = hybrid_r.test_out_acc - learned_r.test_out_acc
            print(f"  {task_name:30s}: "
                  f"in {delta_in:+.4f}, out {delta_out:+.4f}")

    # ===== VERDICT =====
    print(f"\n{'='*70}")
    print("PHASE 3 VERDICT")
    print(f"{'='*70}")

    if spirnor_total > 0 and spirnor_wins / spirnor_total > 0.5:
        print("\nPOSITIVE: SPIRNOR shows advantage on arithmetic tasks.")
        print("The number-theoretic structure in SPIRNOR embeddings helps")
        print("transformers learn and generalize on multiplicative tasks.")
    elif spirnor_total > 0 and spirnor_wins / spirnor_total > 0.25:
        print("\nMIXED: SPIRNOR shows advantage on SOME arithmetic tasks.")
        print("The structure helps for specific task types but not universally.")
    else:
        print("\nNEGATIVE: SPIRNOR does not show clear advantage.")
        print("The transformer can learn arithmetic structure from data alone,")
        print("or the SPIRNOR embedding doesn't encode it in a useful form.")

    print("\nKey question answered:")
    print("  Does SPIRNOR's multiplicative structure transfer to arithmetic reasoning?")

    # Save results
    results_path = os.path.join(os.path.dirname(__file__),
                                'SPIRNOR_AI_PHASE3_RESULTS.json')
    json_results = []
    for r in all_results:
        json_results.append({
            'task': r.task,
            'embedding': r.embedding,
            'n_params': r.n_params,
            'train_acc': r.train_acc,
            'test_in_acc': r.test_in_acc,
            'test_out_acc': r.test_out_acc,
            'test_in_f1': r.test_in_f1,
            'test_out_f1': r.test_out_f1,
            'train_time': r.train_time,
            'epochs': r.epochs,
            'final_loss': r.final_loss,
        })

    try:
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to: {results_path}")
    except Exception as e:
        print(f"\nCould not save results: {e}")

    return all_results


if __name__ == '__main__':
    run_all()
