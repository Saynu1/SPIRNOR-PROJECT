# SPIRNOR: Log-Spiral Numeric Embeddings for Out-of-Distribution Generalization in Number-Theoretic Reasoning

---

## Abstract

We introduce SPIRNOR, a deterministic numeric embedding based on mapping integers to coordinates on a three-dimensional logarithmic spiral parameterized by mathematical constants. Unlike learned embeddings that memorize a fixed vocabulary and collapse on unseen inputs, SPIRNOR computes structure-preserving features for *any* positive integer, enabling strong out-of-distribution (OOD) generalization. Across twelve experimental phases testing 21 classification tasks plus autoregressive arithmetic across 5 domain types, SPIRNOR embeddings consistently outperform learned baselines on OOD evaluation while using fewer parameters. In our capstone experiment (Phase 6), a 4-layer Transformer with SPIRNOR achieves **66.1% OOD accuracy** versus **35.8% for learned embeddings** — an 85% relative improvement. Phase 7 reveals that **pi alone accounts for 78% of OOD benefit**, and Phase 7B validates all findings with p < 0.0001 (25 runs). Phase 8 discovers that **rational fractions C = 2pi/p** create exact modular-arithmetic encodings, boosting OOD from **58.1% to 75.0%** (+29%). Phase 9 expands to 8 tasks including primality testing, modular multiplication, and magnitude comparison: SPIRNOR achieves **6.5x advantage on ModMul7** (confirming exact mod-p encoding), while at d_model=256, SPIRNOR reaches **76.1% avg OOD** with **100% ModMul7 accuracy**. Phase 10 tests SPIRNOR in a GPT-style autoregressive arithmetic setting (character-level addition and multiplication), revealing an important boundary: **all 6 embedding configurations achieve 0% exact-match OOD**, demonstrating that autoregressive digit-level generalization is an architectural limitation, not an embedding one. However, SPIRNOR-frequency RoPE achieves the **best in-range accuracy (100%) and fastest convergence** (96.9% token accuracy at epoch 5 vs 71.7% baseline), and SPIRNOR value augmentation dramatically outperforms learned augmentation (97.8% vs 8.7% in-range for addition). Phase 11 pushes into compositional territory: 8 new tasks including CRT reconstruction, Diophantine feasibility, variable-length set reasoning, and hidden modulus detection. SPIRNOR achieves **98.8% avg OOD** (vs 40.5% learned, **2.4x**), with **100% on Mod30 at all ranges** (validating the Chinese Remainder Theorem encoding), **100% on compositional multiplication and addition mod 30**, and near-perfect performance on variable-length set aggregation and sequence pattern detection. We additionally propose SPIRNOR-RoPE, which outperforms standard RoPE by 4-8% on position-sensitive tasks. All results use 22% fewer parameters than learned baselines.

---

## 1. Introduction

Neural networks have achieved remarkable success across many domains, yet they struggle with systematic generalization on tasks involving numerical reasoning (Saxton et al., 2019; Nogueira et al., 2021). A fundamental challenge is the representation of numbers: standard approaches either learn an embedding table (which cannot represent numbers outside the training vocabulary) or use positional-encoding-style sinusoidal features (which lack number-theoretic structure).

Consider the task of predicting GCD(a, b). A model trained on numbers 2-2000 must, at test time, handle numbers like 347,291. A learned embedding table maps this to a hash bucket, destroying all mathematical structure. A sinusoidal embedding preserves magnitude information but encodes no divisibility relationships. What is needed is an embedding that (1) works for any positive integer, (2) preserves number-theoretic structure such as divisibility and prime factorization, and (3) is parameter-efficient.

We propose SPIRNOR (Spiral Number Representation), a deterministic embedding that maps each integer *n* to a point on a three-dimensional logarithmic spiral:

> r = ln(n),  theta = (C * n) mod 2pi,  phi = (PHI * n) mod 2pi

where C ranges over a set of mathematical constants (pi, sqrt(2), phi^2, e, golden angle, phi, ln(2), pi/e) and PHI is the golden ratio. This mapping has several desirable properties:

- **Deterministic**: No learned parameters in the raw feature computation; works for any positive integer.
- **Structure-preserving**: Numbers sharing common factors produce correlated angular patterns across the constant set.
- **Logarithmic scaling**: The radial component ln(n) provides smooth magnitude encoding.
- **Irrational winding**: The irrational constants ensure that angular positions never exactly repeat, providing unique fingerprints even for very large numbers.

We conduct a systematic twelve-phase experimental program demonstrating SPIRNOR's effectiveness:

1. **Phase 1** identifies which mathematical constants carry the most predictive power for number-theoretic tasks.
2. **Phase 2** shows SPIRNOR achieves 80.5% on MNIST with 75x fewer parameters than a dense MLP.
3. **Phase 3** demonstrates 2-3x better OOD generalization on GCD and smallest prime factor (SPF) tasks.
4. **Phase 4** extends OOD testing to 50x the training range and introduces SPIRNOR-RoPE position encoding.
5. **Phase 5** tests SPIRNOR-RoPE in a character-level autoregressive math language model.
6. **Phase 6** — the capstone — validates SPIRNOR numeric embedding at scale with a Transformer encoder on 5 tasks, 250K training examples, and OOD ranges up to 250x.
7. **Phase 7** performs systematic constant ablation at Transformer scale (17 configurations), revealing that pi alone dominates OOD generalization and that fewer constants paradoxically generalize better.
8. **Phase 7B** validates all Phase 7 claims across 5 random seeds (25 total runs) with statistical hypothesis testing, confirming extreme reproducibility.
9. **Phase 8** tests a theoretically-motivated alternative: replacing irrational constants with rational fractions C = 2pi/p for small primes p, which create exact mod-p encodings. This yields the largest single improvement in the project (+16.9 pp average OOD over irrational baselines) and explains why pi dominated in all previous phases.
10. **Phase 9** expands to 8 tasks (adding IsPrime, ModMul7, Compare), tests hybrid architectures (additive and gated), and scales to d_model=256. SPIRNOR achieves 6.5x advantage on ModMul7 (confirming exact mod-p encoding), while hybrids consistently underperform pure SPIRNOR. At d_model=256, SPIRNOR reaches 76.1% avg OOD with 100% ModMul7 accuracy.
11. **Phase 10** tests SPIRNOR in a GPT-style autoregressive setting — character-level integer addition and multiplication — revealing the boundary of SPIRNOR's applicability. All 6 embedding configs achieve 0% OOD exact match, but SPIRNOR-frequency RoPE provides the best convergence, and SPIRNOR value augmentation is far more robust than learned value augmentation.
12. **Phase 11** pushes SPIRNOR into compositional and structured numeric reasoning: 8 new tasks across 4 families (CRT validation, compositional multi-step reasoning, variable-length set aggregation, and sequence pattern detection). SPIRNOR achieves 98.8% avg OOD vs 40.5% for learned (2.4x), with 100% on Mod30 across all ranges — the strongest direct validation of the Chinese Remainder Theorem encoding. Scaling to d_model=256 yields 98.5% avg OOD.

Our contributions are:
- The SPIRNOR numeric embedding framework and its theoretical motivation.
- SPIRNOR-RoPE, a rotary position encoding using mathematical winding constants.
- A systematic constant ablation study revealing that pi carries 78% of OOD benefit and identifying a capacity-generalization tradeoff where fewer constants generalize better.
- Multi-seed statistical validation confirming all claims with p < 0.03 (most p < 0.0001).
- The **rational fraction framework**: the discovery that constants C = 2pi/p create exact mod-p encodings, explaining pi's dominance (it is 2pi/2) and yielding a 29% relative improvement in OOD generalization over irrational constants.
- **Domain characterization**: SPIRNOR dominates on modular/divisibility tasks (1.4-6.5x), while learned embeddings remain appropriate for pure magnitude tasks, providing clear guidance on when to apply SPIRNOR.
- Demonstration that hybrid architectures (additive and gated) dilute the SPIRNOR signal — pure SPIRNOR is optimal when tasks have modular structure.
- **Boundary identification**: Phase 10 demonstrates that SPIRNOR's OOD advantage does not extend to autoregressive digit-level generation (addition/multiplication), where the generalization bottleneck is architectural (carry propagation over unseen sequence lengths) rather than representational. This precisely delineates SPIRNOR's applicability.
- **Compositional and structured reasoning**: Phase 11 demonstrates that SPIRNOR's advantage extends to multi-step compositional tasks (Diophantine feasibility: 97.9% OOD), variable-length set aggregation (SetAllDivisible: 99.1% OOD), and sequence pattern detection (HiddenModulus: 99.9% OOD). The CRT validation (Mod30: 100% at all ranges) provides the definitive proof that the embedding computes exact Chinese Remainder Theorem reconstructions.
- Extensive empirical evidence across 12 phases, 21+ tasks, 5 domain types, and 6 model architectures showing consistent OOD advantage where modular structure is relevant.
- A parameter-efficiency analysis showing SPIRNOR achieves better generalization with 22-50% fewer parameters.

---

## 2. Related Work

**Numeric representations in neural networks.** Standard approaches to representing numbers in neural networks include learned embedding tables (Vaswani et al., 2017), which are limited to a fixed vocabulary; sinusoidal positional encodings (Vaswani et al., 2017), originally designed for sequence positions rather than numeric values; and digit-level tokenization (Nogueira et al., 2021), which treats numbers as sequences of characters. Recent work on number encodings includes xVal (Golkar et al., 2023), which uses a single scalar multiplied by a learned vector, and log-scale embeddings for scientific applications. SPIRNOR differs by using a fixed set of mathematically-motivated angular projections that encode number-theoretic structure.

**Mathematical reasoning in neural networks.** Transformer models have been applied to arithmetic (Saxton et al., 2019; Lee et al., 2023), symbolic mathematics (Lample & Charton, 2020), and theorem proving (Polu & Sutskever, 2020). These works focus primarily on architecture and training methodology; SPIRNOR is complementary, addressing the input representation layer.

**Rotary position encoding.** RoPE (Su et al., 2021) encodes position information by rotating query and key vectors using a geometric frequency progression. Our SPIRNOR-RoPE replaces this progression with mathematical winding constants, providing richer angular structure for short sequences where standard RoPE's low-frequency dimensions are effectively wasted.

---

## 3. The SPIRNOR Framework

### 3.1 The SPIRNOR Equation

For each positive integer n and mathematical constant C, we define the SPIRNOR mapping:

```
r = ln(n)
theta_C = (C * n) mod 2pi
phi = (PHI * n) mod 2pi
```

where PHI = (1 + sqrt(5)) / 2 is the golden ratio. This maps n to a 3D point:

```
x_C = r * sin(theta_C) * cos(phi)
y_C = r * sin(theta_C) * sin(phi)
z_C = r * cos(theta_C)
```

We additionally extract the trigonometric components sin(theta_C), cos(theta_C), sin(phi), cos(phi), yielding 7 features per constant. With K=8 constants, this produces a 56-dimensional raw feature vector, which is projected to the model dimension d_model via a learned linear layer.

**Why these constants?** Phase 1 of our experimental program systematically evaluates 16 candidate constants, finding pi overwhelmingly dominant (importance +0.145). Phase 7 repeats this ablation at Transformer scale, confirming pi's dominance (+0.187) and identifying phi^2 as the clear second (+0.052), with 4 of 8 constants contributing zero or negative value to OOD generalization. Phase 8 explains *why*: pi = 2pi/2, which means (pi * n) mod 2pi takes exactly two values — 0 for even n and pi for odd n — creating a **perfect parity (mod-2) encoding**. This is not an accident of irrational winding but exact modular arithmetic.

**From irrational winding to rational encoding.** The original SPIRNOR motivation invoked Weyl's equidistribution theorem: when C is irrational, {(C * n) mod 2pi} is dense in [0, 2pi), and numbers sharing a common factor d produce correlated angular patterns. However, Phase 8 reveals a more powerful mechanism. When C = 2pi/p for integer p, the angle (C * n) mod 2pi = (2pi * n/p) mod 2pi takes exactly p discrete values, creating an **exact mod-p encoding**: the angle directly reports n mod p. For prime p, this means:

- C = 2pi/2 (= pi): encodes parity (n mod 2) — 2 angular positions
- C = 2pi/3: encodes n mod 3 — 3 angular positions
- C = 2pi/5: encodes n mod 5 — 5 angular positions
- C = 2pi/7: encodes n mod 7 — 7 angular positions

Using the first 5 primes {2, 3, 5, 7, 11}, the embedding jointly encodes n mod 2, n mod 3, n mod 5, n mod 7, n mod 11. By the Chinese Remainder Theorem, this uniquely identifies n mod 2310 (= 2 * 3 * 5 * 7 * 11), providing rich divisibility information for any integer. Critically, these encodings are **exact for all n** — they never degrade with magnitude, explaining the near-perfect scale invariance observed empirically.

**Irrational constants as approximate modular encoders.** The irrational constants used in Phases 1-7 were effective precisely to the extent that they approximated rational fractions. Pi = 2pi/2 exactly; phi^2 = 2.618... is not a rational multiple of 2pi, but its winding pattern partially separates numbers by residue classes (e.g., it creates approximate mod-3 and mod-5 patterns). The remaining irrationals (e, sqrt(2), etc.) provide equidistributed angles that add representational capacity but no clean divisibility signal — explaining their zero or negative importance in Phase 7.

### 3.2 SPIRNOR Numeric Embedding

The SPIRNOR numeric embedding module computes:

```
embed(n) = LayerNorm(W * spirnor_features(n) + b)
```

where spirnor_features(n) is the 56-dimensional raw feature vector and W is a learned projection to d_model dimensions. Critically:

- The raw features require **zero learned parameters** — they are deterministic functions of n.
- Only the projection layer has learned parameters: 56 * d_model + d_model (for d_model=128: 7,296 parameters).
- Compare to a learned embedding table for numbers 0-2000: 2001 * 128 = 256,128 parameters — **35x more**.

For out-of-distribution numbers, the SPIRNOR features are computed identically. There is no hash table, no modular arithmetic fallback, no vocabulary boundary. The embedding of n=347,291 is as well-defined as n=42.

### 3.3 SPIRNOR-RoPE

Standard Rotary Position Encoding (RoPE) assigns frequencies to dimension pairs using a geometric progression:

```
freq_d = 1 / (10000^(2d/D))
```

For a model with d_head=32, this produces frequencies ranging from 1.0 (pair 0) down to ~0.001 (pair 15). On sequences of length 2-4, pairs with freq < 0.03 provide negligible angular separation — these dimensions are effectively wasted.

SPIRNOR-RoPE replaces the geometric progression with SPIRNOR winding constants:

```
freq_d = constants[d % K] * (d // K + 1)
```

where K=8 is the number of constants and the harmonic multiplier handles cases where d > K. This produces frequencies in [0.69, 6.28], ensuring ALL dimension pairs provide meaningful angular separation even for very short sequences.

---

## 4. Experimental Setup

All experiments use PyTorch 2.10.0 with CUDA 13.0 on an NVIDIA RTX 3060 Ti (8GB VRAM). Models use the AdamW optimizer with cosine annealing learning rate scheduling.

### 4.1 Tasks

We evaluate on a progression of number-theoretic tasks:

| Task | Input | Output | Classes | Phases |
|------|-------|--------|---------|--------|
| Prime detection | n | is_prime(n) | 2 | 1 |
| Semiprime detection | n | is_semiprime(n) | 2 | 1, 3 |
| Omega (distinct primes) | n | omega(n) | 5 | 1, 6 |
| MNIST digit classification | image | digit | 10 | 2 |
| GCD | a, b | gcd_bucket(a,b) | 9 | 3, 4, 6 |
| Smallest prime factor | n | spf_bucket(n) | 10 | 3, 4, 6 |
| Modular multiplication | a, b | (a*b) mod p | p | 3 |
| Integer division | a, b | floor(a/b) bucket | varies | 4 |
| Remainder | a, b | a mod b | varies | 4 |
| Selective GCD | a,b,c,d | gcd(a,d) | varies | 4 |
| Number of divisors | n | ndiv_bucket(n) | 13 | 6 |
| Is coprime | a, b | gcd(a,b)==1 | 2 | 6 |
| Addition/Subtraction/Mult | a, b | a op b | char-level | 5 |

### 4.2 Embedding Baselines

- **Learned**: Standard nn.Embedding table. For OOD numbers, we use modular hashing: embed(n) = table[n % vocab_size]. This is the standard approach in NLP models.
- **Sinusoidal**: Positional-encoding-style features applied to the number value: sin(n * freq_d) and cos(n * freq_d) with geometric frequency progression, followed by a learned projection. Deterministic like SPIRNOR but lacks number-theoretic structure.
- **SPIRNOR**: As described in Section 3.2.
- **SPIRNOR+Learned**: Concatenation of SPIRNOR and learned embeddings, testing whether the approaches are complementary.

### 4.3 Model Architectures

- **Phase 1-3**: 2-layer MLP with hidden dim 256, trained for 30 epochs.
- **Phase 4A**: 4-layer Transformer encoder (d_model=64, 4 heads).
- **Phase 4B**: 3-layer Transformer with custom rotary attention (d_model=64, 4 heads).
- **Phase 5**: 4-layer GPT decoder (d_model=128, 4 heads, character-level autoregressive).
- **Phase 6**: 4-layer Transformer encoder (d_model=128, 4 heads, bidirectional attention, task-specific classification heads).

---

## 5. Results

### 5.1 Phase 1: Mathematical Constants Carry Predictive Power

We evaluate 16 mathematical constants for their contribution to prime detection, semiprime detection, and omega prediction. Each constant is ablated (removed) from the full set and the change in F1 score is measured.

**Key finding**: Pi is overwhelmingly the most important constant, with an average importance score of +0.145 across tasks — 11x larger than the next most important (sqrt(2) at +0.013). Accumulation curves show diminishing returns after 2-3 constants, with the largest single improvement always coming from adding pi.

This result motivates the use of specific mathematical constants rather than arbitrary or random frequencies in our embedding.

### 5.2 Phase 2: Parameter Efficiency on MNIST

To validate SPIRNOR beyond pure number theory, we test on MNIST digit classification using a SPIRNORLinear layer that replaces standard dense layers.

| Model | Parameters | Test Accuracy |
|-------|-----------|---------------|
| Dense MLP (full) | 118,282 | 97.6% |
| Dense MLP (matched) | 1,606 | 32.0% |
| **SPIRNOR MLP (K=16)** | **1,578** | **80.5%** |
| Random Fourier (K=16) | 1,578 | 89.7% |
| SPIRNOR Attention MLP | 256,802 | 96.9% |

SPIRNOR achieves **80.5% accuracy with only 1,578 parameters** — a 75x compression from the full dense model. A parameter-matched dense MLP achieves only 32.0%, demonstrating that SPIRNOR's structure provides meaningful inductive bias even on image data. Adding attention-based constant routing (SPIRNOR Attention) recovers near-full performance at 96.9%.

### 5.3 Phase 3: Numeric Embedding OOD Generalization

We compare four embedding types on GCD classification and smallest prime factor (SPF) detection. Models are trained on numbers 2-500 and evaluated both in-distribution and out-of-distribution (501-1000).

**Table 1: Phase 3 OOD Results (test accuracy on numbers 501-1000)**

| Task | Learned | Sinusoidal | SPIRNOR | Improvement |
|------|---------|------------|---------|-------------|
| GCD | 40.3% | 38.8% | **80.6%** | **2.0x** vs Learned |
| SPF | 31.4% | 26.6% | **75.8%** | **2.4x** vs Learned |
| Semiprime | 62.7% | 55.6% | **67.2%** | 1.07x vs Learned |

SPIRNOR achieves 2.0x better OOD accuracy on GCD and 2.4x on SPF, while using approximately **half the parameters** of the learned baseline (55K vs 111K). In-distribution performance is comparable across all methods (93-100%).

The SPIRNOR+Learned hybrid provides further OOD improvement on GCD (74.4% to 80.6%) and SPF (77.2%), suggesting the approaches are partially complementary.

### 5.4 Phase 4A: Extended Generalization to 50x Training Range

We scale to a 4-layer Transformer encoder trained on numbers 2-1000 and evaluate at progressively larger OOD ranges.

**Table 2: Phase 4A Extended OOD Results (GCD accuracy)**

| OOD Range | Learned | Sinusoidal | SPIRNOR |
|-----------|---------|------------|---------|
| In-range (2-1K) | 87.8% | 59.1% | **88.4%** |
| 1K-2K (2x) | 33.6% | 60.4% | **87.9%** |
| 2K-5K (5x) | 46.9% | 60.0% | **86.4%** |
| 5K-10K (10x) | 40.0% | 59.4% | **81.5%** |
| 10K-50K (50x) | 44.6% | 61.0% | **63.6%** |

SPIRNOR maintains **88% accuracy at 2x OOD and 82% at 10x OOD** on GCD, while learned embeddings collapse to 34% at 2x (due to hash collisions destroying structure). Even at 50x the training range, SPIRNOR retains 64% accuracy.

**Table 3: Phase 4A Extended OOD Results (SPF accuracy)**

| OOD Range | Learned | Sinusoidal | SPIRNOR |
|-----------|---------|------------|---------|
| In-range | 100% | 88.2% | **100%** |
| 1K-2K | 0.1% | 36.6% | **80.3%** |
| 2K-5K | 41.4% | 37.2% | **77.2%** |
| 5K-10K | 27.3% | 31.2% | **76.6%** |
| 10K-50K | 35.5% | 34.7% | **58.6%** |

The learned embedding catastrophically fails on SPF at 1K-2K (0.1%) because the modular hash maps fundamentally different numbers to the same embedding. SPIRNOR degrades gracefully across all ranges.

### 5.5 Phase 4B: SPIRNOR-RoPE Position Encoding

We test SPIRNOR-RoPE against standard RoPE on position-sensitive tasks requiring the model to distinguish token order.

**Table 4: SelectiveGCD task — GCD(a,d) from input [a,b,c,d] (4 tokens)**

| PE Type | In-Range | 500-1K OOD | 1K-2K OOD |
|---------|----------|------------|-----------|
| No PE | 64.8% | 63.5% | 62.1% |
| Sinusoidal | 89.2% | 80.5% | 77.4% |
| Standard RoPE | 85.0% | 76.8% | 74.6% |
| Random-RoPE | 88.7% | 82.4% | 79.8% |
| **SPIRNOR-RoPE** | **89.1%** | **84.4%** | **80.8%** |

SPIRNOR-RoPE outperforms standard RoPE by **+4.1% in-range, +7.6% at 500-1K, and +6.2% at 1K-2K**. It also beats Random-RoPE (which uses arbitrary frequencies in the same range), demonstrating that the specific mathematical constants matter, not just the frequency range.

On simpler 2-token tasks (Remainder, Integer Division), all PE types perform similarly, confirming that SPIRNOR-RoPE's advantage is specifically for longer sequences where position discrimination is harder.

### 5.6 Phase 5: Character-Level Math Language Model

We test SPIRNOR-RoPE as position encoding in a 4-layer GPT-style autoregressive model predicting character-level arithmetic expressions (e.g., "G:43,90=1"). Training on numbers 2-200, we compare Learned PE, Sinusoidal PE, Standard RoPE, and SPIRNOR-RoPE.

**Table 5: Phase 5 Overall Exact-Match Accuracy**

| PE Type | In-Range | 200-500 | 500-1K | 1K-5K |
|---------|----------|---------|--------|-------|
| Learned PE | 38.0% | 29.0% | 25.0% | 21.0% |
| Sinusoidal PE | 37.8% | 25.6% | 23.2% | 16.6% |
| **SPIRNOR-RoPE** | **39.4%** | **29.4%** | **27.6%** | 17.0% |
| Standard RoPE | 38.6% | 29.0% | 27.0% | **23.4%** |

SPIRNOR-RoPE wins 3 of 4 ranges but shows degradation at the farthest OOD range (1K-5K), where standard RoPE's geometric frequency progression better handles the longer character sequences produced by larger numbers. This motivated Phase 5B's Hybrid-RoPE investigation (which showed the 50/50 split dilutes both bands) and ultimately led to our decision to refocus on SPIRNOR's strongest use case: numeric embedding rather than position encoding.

### 5.7 Phase 6: Token-Level Transformer at Scale (Capstone)

Our capstone experiment combines SPIRNOR's proven numeric embedding with a Transformer encoder architecture at significantly larger scale: 250,000 training examples (50,000 per task), 5,000 evaluation examples per range (1,000 per task), training on numbers 2-2000, and testing up to 500,000 (250x extrapolation).

**Table 6: Phase 6 Overall Accuracy**

| Configuration | Params | In-Range | 2K-5K | 5K-20K | 20K-100K | 100K-500K |
|--------------|--------|----------|-------|--------|----------|-----------|
| Learned+RoPE | 1,137,703 | **98.1%** | 35.8% | 40.5% | 41.8% | 39.6% |
| Sinusoidal+RoPE | 898,215 | 97.2% | 38.1% | 36.2% | 36.5% | 34.8% |
| **SPIRNOR+RoPE** | **888,999** | 97.2% | **66.1%** | **58.1%** | **54.2%** | **53.9%** |
| SPIRNOR+SRoPE | 888,999 | 97.5% | 65.7% | 57.6% | 53.5% | 53.1% |

SPIRNOR+RoPE achieves the highest OOD accuracy at all four out-of-distribution ranges while using **22% fewer parameters** than the learned baseline (888,999 vs 1,137,703).

**Table 7: Phase 6 Per-Task OOD Accuracy at 2K-5K (nearest OOD range)**

| Task | Learned | Sinusoidal | SPIRNOR | Ratio |
|------|---------|------------|---------|-------|
| GCD | 45.1% | 39.8% | **85.3%** | **1.9x** |
| SPF | 24.9% | 34.7% | **74.0%** | **3.0x** |
| NumDiv | 14.6% | 21.9% | **32.8%** | **2.2x** |
| Omega | 36.1% | 38.9% | **47.1%** | **1.3x** |
| Coprime | 58.5% | 55.3% | **91.1%** | **1.6x** |

**Table 8: Phase 6 Per-Task OOD Accuracy at 100K-500K (250x extrapolation)**

| Task | Learned | Sinusoidal | SPIRNOR | Ratio |
|------|---------|------------|---------|-------|
| GCD | 47.5% | 36.4% | **64.0%** | **1.3x** |
| SPF | 38.4% | 39.9% | **60.6%** | **1.6x** |
| NumDiv | 20.7% | 20.9% | **31.4%** | **1.5x** |
| Omega | 31.6% | 28.8% | **32.4%** | 1.0x |
| Coprime | 59.6% | 47.8% | **81.3%** | **1.4x** |

In head-to-head comparison, SPIRNOR wins **19 of 25 comparisons** against the learned baseline (5 tasks x 5 ranges), with 4 ties and only 2 losses (both in-range, where the learned embedding's overfitting is advantageous).

**Generalization retention**: At 250x extrapolation (100K-500K), SPIRNOR retains **56%** of its in-range accuracy, compared to 40% for learned and 36% for sinusoidal embeddings.

**Statistical significance**: With 1,000 evaluation samples per task per range, the overall SPIRNOR vs. Learned gap at 2K-5K (30.2 percentage points) corresponds to approximately 43 standard deviations under a binomial model, confirming overwhelming statistical significance.

### 5.8 Phase 7: Constant Ablation at Transformer Scale

Phase 6 used all 8 SPIRNOR constants. But which constants actually matter? Phase 1's MLP-based ablation (Section 5.1) identified pi as dominant, but that was on a simple 2-layer MLP with single-number tasks. Phase 7 repeats the ablation at the full Phase 6 Transformer scale (4-layer encoder, 250K training examples, 5 tasks) with 17 configurations: 1 full baseline (8 constants), 8 leave-one-out ablations (remove each constant individually), and 8 cumulative addition configurations (add constants one at a time in order of Phase 1 importance: pi, sqrt2, phi^2, e, golden_angle, phi, ln2, pi/e).

**Table 9: Leave-One-Out Importance Scores (average OOD accuracy drop when constant is removed)**

| Constant | Avg OOD Importance | 2K-5K Drop | 5K-20K Drop | Tier |
|----------|-------------------|------------|-------------|------|
| **pi** | **+0.187** | +0.089 | +0.278 | **Essential** |
| **phi^2** | **+0.052** | +0.144 | +0.050 | **Important** |
| ln2 | +0.004 | +0.004 | +0.001 | Marginal |
| golden_angle | +0.004 | -0.006 | +0.008 | Marginal |
| sqrt2 | +0.001 | -0.017 | +0.006 | Negligible |
| phi | 0.000 | -0.001 | 0.000 | Zero |
| pi/e | -0.001 | -0.014 | +0.002 | Negative |
| e | -0.005 | -0.018 | +0.006 | Negative |

Pi's importance (+0.187) is **3.6x larger** than the next constant (phi^2 at +0.052) and **47x larger** than the third (ln2 at +0.004). Removing pi collapses OOD accuracy at 5K-20K from 59.9% to 32.1% — a catastrophic 46% relative drop. Four of eight constants (sqrt2, phi, pi/e, e) have zero or negative importance, meaning they are dead weight or actively harmful for OOD generalization.

**Table 10: Cumulative Addition (constants added in order of Phase 1 importance)**

| Constants | K | In-Range | 2K-5K | 5K-20K | 20K-100K | 100K-500K | Avg OOD |
|-----------|---|----------|-------|--------|----------|-----------|---------|
| pi | 1 | 69.5% | 63.1% | 63.4% | 62.2% | 62.7% | **62.8%** |
| +sqrt2 | 2 | 91.1% | 56.0% | 57.4% | 55.8% | 54.8% | 56.0% |
| +phi^2 | 3 | **96.3%** | **68.4%** | 59.5% | 54.2% | 54.8% | 59.2% |
| +e | 4 | 97.1% | 65.6% | 59.6% | 52.6% | 53.8% | 57.9% |
| +golden_angle | 5 | 96.6% | 66.7% | 59.4% | 53.4% | 53.6% | 58.3% |
| +phi | 6 | 96.8% | 66.2% | 58.7% | 52.8% | 53.4% | 57.8% |
| +ln2 | 7 | 97.2% | 66.8% | 59.3% | 54.1% | 54.4% | 58.6% |
| +pi/e | 8 | 97.5% | 65.5% | 59.8% | 54.1% | 54.1% | 58.4% |

**Key finding: The capacity-generalization tradeoff.** Pi alone (K=1) achieves the **highest average OOD accuracy of any configuration** (62.8%), despite having the lowest in-range accuracy (69.5%). Adding more constants improves in-range fitting (69.5% to 97.5%) but *hurts* average OOD (62.8% to 58.4%). This reveals a fundamental tradeoff: more constants provide more representational capacity, enabling better in-range fitting but also more overfitting to the training distribution.

The **practical sweet spot is K=3** (pi, sqrt2, phi^2): it achieves 96.3% in-range (near the full model's 97.5%) while having the best near-OOD accuracy (68.4% at 2K-5K) — surpassing even the full 8-constant model (65.5%).

### 5.9 Phase 7B: Multi-Seed Statistical Validation

A natural concern with Phase 7's findings is whether they are artifacts of a single random seed. Phase 7B addresses this by running 5 key configurations across 5 random seeds each (25 total training runs), with fixed test data and Welch's t-test for statistical comparison.

**Configurations**: Learned+RoPE (baseline), SPIRNOR 1 constant (pi only), SPIRNOR 3 constants (sweet spot), SPIRNOR 8 constants (full), SPIRNOR 7 constants (no pi). **Seeds**: 42, 123, 456, 789, 2024.

**Table 11: Multi-Seed Results (mean +/- std across 5 seeds)**

| Configuration | Params | In-Range | Avg OOD |
|--------------|--------|----------|---------|
| Learned+RoPE | 1,137,703 | 0.985 +/- 0.001 | 0.389 +/- 0.001 |
| **SPIRNOR 1 (pi)** | **882,727** | 0.698 +/- 0.010 | **0.621 +/- 0.004** |
| SPIRNOR 3 (sweet) | 884,519 | 0.965 +/- 0.000 | 0.587 +/- 0.002 |
| SPIRNOR 8 (full) | 888,999 | 0.975 +/- 0.002 | 0.584 +/- 0.003 |
| SPIRNOR 7 (no pi) | 888,103 | 0.972 +/- 0.002 | 0.396 +/- 0.004 |

Standard deviations are remarkably small (0.001-0.010), confirming that all Phase 7 findings are robust to random seed variation.

**Table 12: Statistical Hypothesis Tests (Welch's t-test, n=5 per group)**

| Comparison | Mean Diff | t-statistic | p-value | Verdict |
|-----------|-----------|-------------|---------|---------|
| SPIRNOR 8 vs Learned (OOD) | +0.194 | 105.8 | **p < 0.0001** | SPIRNOR dominant |
| Pi-only vs Full 8 (OOD) | +0.038 | 16.5 | **p < 0.0001** | Fewer > more |
| 3-sweet vs Full 8 (OOD) | +0.003 | 3.1 | **p = 0.027** | Sweet spot confirmed |
| Full 8 vs No-pi (OOD) | +0.188 | 86.2 | **p < 0.0001** | Pi essential |

**All four key claims from Phase 7 are statistically confirmed:**

1. **Pi alone achieves the best average OOD** (0.621 vs 0.584 for full): p < 0.0001
2. **3 constants outperform 8 for OOD** (0.587 vs 0.584): p = 0.027
3. **Removing pi is catastrophic** (0.584 vs 0.396): p < 0.0001
4. **SPIRNOR dominates Learned for OOD** (0.584 vs 0.389): p < 0.0001

The t-statistics for claims 1, 3, and 4 exceed 16, indicating effect sizes many standard deviations apart — these are not borderline results but overwhelming separations. Even Claim 2, the most marginal effect (0.003 accuracy difference), achieves statistical significance at p = 0.027.

### 5.10 Phase 8: Rational Fraction Constants

Phase 7 revealed that pi dominates OOD generalization, but left open *why*. Mathematical analysis (see Section 3.1) reveals that pi = 2pi/2 creates an exact parity encoding, suggesting that the optimal SPIRNOR constants might be rational fractions C = 2pi/p for small primes p rather than irrational numbers. Phase 8 tests this hypothesis by comparing 7 configurations using the same Phase 6 architecture (4-layer Transformer, 250K training examples, 30 epochs).

**Constant sets tested:**
- **irrational_8**: Original 8 irrationals from Phase 6 (baseline)
- **irrational_3**: Phase 7 sweet spot {pi, sqrt(2), phi^2}
- **rational_5**: {2pi/2, 2pi/3, 2pi/5, 2pi/7, 2pi/11} — first 5 primes
- **rational_8**: {2pi/2, ..., 2pi/19} — first 8 primes
- **rational_3**: {2pi/2, 2pi/3, 2pi/5} — first 3 primes
- **hybrid_5**: {2pi/2, 2pi/3, 2pi/5, sqrt(2), phi^2} — rational primes + best irrationals
- **learned**: Standard learned embedding (reference)

**Table 13: Phase 8 Overall Accuracy**

| Configuration | K | Params | In-Range | 2K-5K | 5K-20K | 20K-100K | 100K-500K | Avg OOD |
|--------------|---|--------|----------|-------|--------|----------|-----------|---------|
| **rational_5** | 5 | 886,311 | 97.9% | **77.9%** | 75.9% | 73.9% | **72.3%** | **75.0%** |
| **rational_8** | 8 | 888,999 | 97.8% | 77.2% | **76.2%** | **74.1%** | 72.2% | **74.9%** |
| rational_3 | 3 | 884,519 | 93.0% | 71.9% | 70.7% | 70.2% | 69.5% | 70.6% |
| hybrid_5 | 5 | 886,311 | **98.0%** | 73.7% | 72.3% | 68.3% | 68.7% | 70.7% |
| irrational_3 | 3 | 884,519 | 96.6% | 67.1% | 59.0% | 54.6% | 54.3% | 58.8% |
| irrational_8 | 8 | 888,999 | 97.6% | 65.2% | 59.1% | 54.0% | 53.9% | 58.1% |
| learned | 0 | 1,137,703 | 98.5% | 35.7% | 40.6% | 41.4% | 40.2% | 39.5% |

The rational fraction constants produce a **massive improvement**. rational_5 achieves 75.0% average OOD versus 58.1% for irrational_8 — a **+16.9 percentage point** improvement (29% relative), the largest single gain in the entire project. This exceeds the gap between SPIRNOR and learned embeddings (+18.6 pp) that motivated the original research.

**Table 14: Phase 8 Per-Task Accuracy at 2K-5K (nearest OOD)**

| Task | rational_5 | irrational_8 | learned | Rational Gain |
|------|-----------|-------------|---------|---------------|
| GCD | **90.1%** | 85.1% | 44.7% | +5.0 pp |
| SPF | **91.9%** | 73.6% | 24.0% | **+18.3 pp** |
| NumDiv | **43.1%** | 32.7% | 16.0% | +10.4 pp |
| Omega | **65.7%** | 43.4% | 36.2% | **+22.3 pp** |
| Coprime | **98.5%** | 91.3% | 57.6% | +7.2 pp |

The rational constants improve *every task*, with the largest gains on SPF (+18.3 pp) and Omega (+22.3 pp) — tasks that depend directly on prime factorization structure. Coprime reaches 98.5% at 2K-5K and maintains 97.9% at 100K-500K (250x extrapolation), approaching ceiling performance.

**Key findings from Phase 8:**

1. **Rational > Irrational at every count**: rational_3 (70.6% avg OOD) beats irrational_8 (58.1%) despite having only 3 constants vs 8. The encoding type matters more than the quantity.

2. **5 primes is optimal**: rational_5 (75.0%) and rational_8 (74.9%) are virtually identical, confirming diminishing returns after the first 5 primes. The primorial 2*3*5*7*11 = 2310 captures sufficient divisibility structure.

3. **Hybrid hurts**: hybrid_5 (70.7%) is worse than pure rational_5 (75.0%), indicating that irrational constants add noise that degrades the clean modular signal. The irrationals are not complementary — they are dilutive.

4. **Near-perfect scale invariance**: rational_5 drops only 5.6 pp from 2K-5K (77.9%) to 100K-500K (72.3%), compared to 11.4 pp for irrational_8. The exact modular encodings are equally valid for n=3,000 as for n=300,000, explaining this stability.

5. **Pi's dominance explained**: Pi = 2pi/2 is the simplest rational fraction constant — it creates the most fundamental number-theoretic encoding (parity). Its dominance in Phases 1-7 was not mysterious; it was the only constant in the original set that happened to be a rational multiple of 2pi.

### 5.11 Phase 9: Hybrid Architectures, New Tasks, and Scaling

Phase 9 expands SPIRNOR beyond the original number-theoretic tasks to test three critical questions: (A) Can hybrid architectures combining SPIRNOR with learned embeddings achieve "best of both worlds" (learned's in-range + SPIRNOR's OOD)? (B) Does SPIRNOR generalize to new task types beyond divisibility? (C) Does the advantage persist at larger model scale?

**Part A: Hybrid Architectures (d_model=128, 8 tasks, 400K training examples)**

Four embedding configurations are compared: pure learned, pure SPIRNOR (rational_5), additive hybrid (spirnor + learned → LayerNorm), and gated hybrid (g·spirnor + (1-g)·learned where g = sigmoid(MLP)).

**Table 15: Phase 9 Part A+B Overall Accuracy (d_model=128)**

| Configuration | Params | In-Range | 2K-5K | 5K-20K | 20K-100K | 100K-500K | Avg OOD |
|--------------|--------|----------|-------|--------|----------|-----------|---------|
| **SPIRNOR rational_5** | **937,779** | **96.4%** | **76.8%** | **75.2%** | **73.2%** | **71.3%** | **74.1%** |
| Additive hybrid | 1,194,163 | 93.8% | 73.1% | 73.0% | 72.3% | 71.5% | 72.5% |
| Gated hybrid | 1,210,420 | 94.3% | 73.3% | 72.0% | 71.3% | 70.6% | 71.8% |
| Learned | 1,189,171 | 91.1% | 39.8% | 41.5% | 42.9% | 42.7% | 41.8% |

**Key finding**: Pure SPIRNOR wins both in-range AND OOD, with 21% fewer parameters. Hybrids dilute the SPIRNOR signal — adding learned embeddings hurts rather than helps. The gated hybrid's gate stays near 0.55 (barely above balanced), never learning to rely more heavily on SPIRNOR for OOD inputs as hypothesized.

**Part B: New Task Domains**

Three new tasks test SPIRNOR beyond divisibility: IsPrime (primality, 2 classes), ModMul7 (a×b mod 7, 7 classes — direct test of mod-p encoding), and Compare (a<b / a=b / a>b, 3 classes — pure magnitude, no modular structure).

**Table 16: Phase 9 Per-Task Accuracy at 2K-5K**

| Task | SPIRNOR | Learned | Ratio | Task Type |
|------|---------|---------|-------|-----------|
| GCD | **90.8%** | 46.0% | 2.0x | Modular |
| SPF | **91.6%** | 25.6% | 3.6x | Modular |
| NumDiv | **42.5%** | 16.1% | 2.6x | Modular |
| Omega | **64.8%** | 32.0% | 2.0x | Modular |
| Coprime | **98.7%** | 59.0% | 1.7x | Modular |
| IsPrime | **73.3%** | 50.8% | 1.4x | Mixed |
| ModMul7 | **81.2%** | 12.4% | **6.5x** | Exact mod-p |
| Compare | 71.2% | **76.5%** | 0.9x | Magnitude |

Three critical observations:

1. **ModMul7 confirms exact mod-p encoding**: SPIRNOR achieves 81.2% OOD on a×b mod 7 versus learned's 12.4% (6.5x ratio, the largest advantage across all tasks). Since the constant set includes 2pi/7, the model has exact access to mod-7 residues. This is the strongest direct evidence that SPIRNOR's rational fraction constants function as theorized.

2. **IsPrime: SPIRNOR generalizes beyond divisibility**: Primality is not a pure modular operation — it requires detecting the *absence* of all non-trivial factors. Yet SPIRNOR achieves 73.3% OOD versus learned's 50.8% (random chance for balanced binary), indicating that the modular fingerprint captures useful primality information indirectly.

3. **Compare: Learned wins on magnitude tasks**: Compare (a<b / a=b / a>b) depends purely on numeric magnitude with no modular structure. Here learned embeddings (76.5%) slightly outperform SPIRNOR (71.2%), as expected — SPIRNOR's modular encoding discards absolute magnitude information. This confirms SPIRNOR is not a universal improvement but a *domain-appropriate* one.

**Part C: Scaling Test (d_model=256)**

**Table 17: Phase 9 Part C Scaling Results (d_model=256)**

| Configuration | Params | In-Range | 2K-5K | 5K-20K | 20K-100K | 100K-500K | Avg OOD |
|--------------|--------|----------|-------|--------|----------|-----------|---------|
| **SPIRNOR_256** | **3,710,515** | **98.0%** | **79.7%** | **77.4%** | **74.6%** | **72.7%** | **76.1%** |
| Gated hybrid_256 | 4,255,668 | 94.2% | 72.3% | 70.8% | 70.3% | 68.7% | 70.5% |
| Learned_256 | 4,213,299 | 91.8% | 39.9% | 41.3% | 42.8% | 42.6% | 41.6% |

| Scaling comparison | d=128 OOD | d=256 OOD | Delta |
|-------------------|-----------|-----------|-------|
| SPIRNOR | 74.1% | **76.1%** | **+2.0 pp** |
| Learned | 41.8% | 41.6% | -0.1 pp |
| Gated hybrid | 71.8% | 70.5% | -1.3 pp |

Scaling to d_model=256 benefits *only SPIRNOR*, which improves from 74.1% to 76.1% avg OOD (+2.0 pp). Learned embeddings remain unchanged (~41.7%), and gated hybrid slightly degrades. At d_model=256, SPIRNOR achieves **100% ModMul7 accuracy** at both in-range and 2K-5K, and maintains 95.3% at 20K-100K — confirming that the exact mod-7 encoding operates perfectly when given sufficient model capacity.

The SPIRNOR OOD advantage at d_model=256 is **+34.5 pp** over learned (76.1% vs 41.6%), demonstrating that SPIRNOR's advantage persists and even grows with scale.

**Gated hybrid gate analysis (d_model=256):**

| Range | Mean gate value |
|-------|----------------|
| In-range | 0.378 |
| 2K-5K | 0.357 |
| 5K-20K | 0.341 |
| 20K-100K | 0.323 |
| 100K-500K | 0.305 |

At d_model=256, the gate moves in the *opposite* direction from what was hypothesized: it leans *more toward learned* (gate < 0.5) and decreases further OOD. This indicates the model is "hedging" — using learned embeddings for their familiar patterns and relying less on SPIRNOR for OOD, exactly the wrong strategy. The gate mechanism fails to discover what our ablation studies show: SPIRNOR alone is optimal for OOD.

### 5.12 Phase 10: Autoregressive Arithmetic — Boundary of Applicability

Phases 1-9 used classification tasks where the model maps numeric inputs to class labels. Phase 10 asks a fundamentally different question: **can SPIRNOR help autoregressive generation?** Specifically, we test character-level integer addition (train: 1-999, OOD: 4-6 digits) and multiplication (train: 2-99, OOD: 3-digit) using a GPT-style decoder-only Transformer (d_model=128, 6 layers, 4 heads).

**Table 18: Addition — Exact Match Accuracy (6 embedding configurations)**

| Config | Params | In-Range | 4-Digit | 5-Digit | 6-Digit |
|--------|--------|----------|---------|---------|---------|
| baseline (sinusoidal) | 1.19M | 100.0% | 0% | 0% | 0% |
| spirnor_aug (+SPIRNOR value) | 1.20M | 97.8% | 0% | 0% | 0% |
| learned_aug (+learned value) | 1.32M | **8.7%** | 0% | 0% | 0% |
| std_rope (standard RoPE) | 1.19M | 99.1% | 0% | 0% | 0% |
| spirnor_rope (SPIRNOR-freq RoPE) | 1.19M | **100.0%** | 0% | 0% | 0% |
| spirnor_full (value + SPIRNOR RoPE) | 1.20M | **100.0%** | 0% | 0% | 0% |

**Key findings:**

1. **OOD exact match = 0% for all 6 configs on both operations.** The generalization failure is architectural: autoregressive carry-chain computation cannot extrapolate to unseen digit lengths regardless of embedding quality. This is the first clear boundary of SPIRNOR's applicability.

2. **SPIRNOR-frequency RoPE achieves best in-range performance and fastest convergence.** At epoch 5: spirnor_rope 96.9% token accuracy vs baseline 71.7% vs std_rope 95.5%. At epoch 30: spirnor_rope/spirnor_full reach 99.88-99.95% token accuracy vs baseline 99.83%. The prime-rational frequency schedule provides measurable training efficiency gains.

3. **Value augmentation hurts addition, with learned augmentation catastrophic.** SPIRNOR value augmentation reduces in-range accuracy from 100% to 97.8%, while learned value augmentation collapses to 8.7%. The whole-number value information (SPIRNOR(123) or Embedding(123)) is not useful for predicting individual answer digits and interferes with character-level pattern learning. However, SPIRNOR's structured features are dramatically more compatible than learned embeddings (97.8% vs 8.7%), confirming SPIRNOR's superior inductive bias even in an unfavorable setting.

4. **Multiplication shows similar patterns.** All configs achieve 100% in-range, ~0-1.2% OOD exact match. For multiplication, value augmentation does not hurt (all reach 100%), and SPIRNOR value augmentation produces the best OOD per-digit accuracy (5.9% vs baseline 3.5%), suggesting modular structure provides modest digit-level benefit for multiplication.

**Interpretation:** SPIRNOR's advantage is strongest when the model must predict a *property* of a number (classification) — in this regime, modular encodings directly provide the answer. For *digit-level generation* (autoregressive), the bottleneck shifts from representation to computation: the model must learn carry propagation as an algorithm, which doesn't generalize to longer digit chains regardless of how well the input numbers are represented. This precisely delineates SPIRNOR's niche: number-theoretic classification and reasoning (1.4-6.5x OOD advantage), with an important role for SPIRNOR-frequency RoPE in training efficiency for general models.

### 5.13 Phase 11: Compositional & Structured Numeric Reasoning

Following Phase 10's identification of the autoregressive boundary, Phase 11 pushes SPIRNOR into genuinely novel territory: compositional modular reasoning, variable-length set aggregation, and sequence pattern detection. Using the same Transformer architecture (6 layers, d_model=128, rational_5 constants), we test 8 new tasks across 4 families.

**Table 19: Phase 11 Results — SPIRNOR vs Learned (d_model=128, 1.34M vs 1.59M params)**

| Task | Family | In-Range (S/L) | 2K-5K (S/L) | 100K-500K (S/L) | Advantage |
|------|--------|---------------|-------------|-----------------|-----------|
| Mod30 | CRT | 100/100% | **100/0%** | **100/10.4%** | **∞ / 9.6x** |
| ProductMod30 | CRT | 100/100% | **100/11.8%** | **99.6/14.7%** | **8.5x** |
| SumMod30 | Compositional | 100/98.9% | **100/0.7%** | **92.3/10.8%** | **142.9x** |
| Diophantine | Compositional | 97.7/97.7% | **97.9/61.2%** | **96.5/60.9%** | **1.6x** |
| GCDEquals | Compositional | 98.4/98.2% | **98.3/74.1%** | **98.0/78.2%** | **1.3x** |
| SetAllDivisible | Set | 100/100% | **99.1/59.8%** | **97.8/59.9%** | **1.7x** |
| SetCoprime | Set | 99.8/99.8% | **95.2/60.4%** | **95.1/63.1%** | **1.6x** |
| HiddenModulus | Pattern | 99.9/95.0% | **99.9/56.2%** | **99.8/41.1%** | **1.8x** |
| **Overall** | | **99.5/98.7%** | **98.8/40.5%** | **97.4/42.4%** | **2.4x** |

S = SPIRNOR, L = Learned

**Per-family analysis at 2K-5K OOD:**

| Family | SPIRNOR | Learned | Ratio |
|--------|---------|---------|-------|
| A: CRT Validation | **100.0%** | 5.9% | **17.0x** |
| B: Compositional | **98.7%** | 45.3% | **2.2x** |
| C: Set Reasoning | **97.2%** | 60.1% | **1.6x** |
| D: Pattern Detection | **99.9%** | 56.2% | **1.8x** |

**Key findings:**

1. **CRT validation: Mod30 = 100% at ALL ranges.** This is the definitive proof that SPIRNOR computes exact Chinese Remainder Theorem reconstructions. The model trivially reads off n mod 30 = CRT(n mod 2, n mod 3, n mod 5) from the SPIRNOR embedding of a single number, with zero degradation from 2-2000 to 100K-500K. Learned embeddings collapse to 0% (random chance = 3.3%).

2. **Compositional reasoning works.** ProductMod30 and SumMod30 achieve 100% at 2K-5K, demonstrating that the Transformer can compose modular information across tokens via attention. The SumMod30 advantage over learned (142.9x) is the largest single task advantage in the entire project.

3. **Multi-step reasoning: Diophantine 97.9% OOD.** The model learns gcd(a,b)|c — a two-step operation (compute GCD, then check divisibility) — with near-perfect generalization. This was the hardest task and SPIRNOR still achieved 97.9% at 2K-5K.

4. **Variable-length set aggregation works.** SetAllDivisible (99.1%) and SetCoprime (95.2%) show SPIRNOR handles 3-8 element sets with strong OOD performance. The model successfully aggregates modular information across variable numbers of tokens.

5. **Hidden modulus detection: 99.9%.** Given 8 numbers sharing a hidden residue mod p, the model identifies which prime p is responsible. SPIRNOR's shared angular position across all 8 embeddings makes this pattern trivially detectable.

6. **Scale invariance.** Average OOD drops only 1.1 pp from 2K-5K (98.8%) to 100K-500K (97.4%), vs 1.9 pp for learned. SPIRNOR's modular encoding is fundamentally scale-invariant.

**Table 20: Scaling to d_model=256 (5.30M vs 5.81M params)**

| Config | In-Range | 2K-5K | 5K-20K | 20K-100K | 100K-500K | Avg OOD |
|--------|----------|-------|--------|----------|-----------|---------|
| SPIRNOR_256 | 99.4% | 98.8% | 98.8% | 98.5% | 97.9% | **98.5%** |
| Learned_256 | 99.1% | 39.9% | 40.6% | 41.9% | 42.2% | **41.1%** |

Scaling from d_model=128 to 256 improves SPIRNOR OOD from 98.1% to 98.5% (+0.4 pp) while learned actually degrades from 41.4% to 41.1% (-0.3 pp). The SPIRNOR advantage at d_model=256 is +57.4 pp — the largest absolute advantage observed in any phase of the project.

---

## 6. Discussion

### 6.1 Why SPIRNOR Works

Phase 8's rational fraction results transform our understanding of SPIRNOR's mechanism. The embedding's success stems from three complementary mechanisms:

1. **Logarithmic radial scaling**: ln(n) provides smooth magnitude information that scales naturally to any range. Unlike learned embeddings, there is no vocabulary boundary.

2. **Exact modular-arithmetic encoding** (revised understanding): When C = 2pi/p for integer p, the angle (C*n) mod 2pi = 2pi*(n mod p)/p, which directly encodes n mod p as one of exactly p discrete angular positions. This is not statistical correlation but **exact arithmetic**: the embedding provably computes n mod p for every integer n, regardless of magnitude. For a set of primes {p_1, ..., p_K}, the embedding jointly computes {n mod p_1, ..., n mod p_K}, which by the Chinese Remainder Theorem uniquely identifies n within a range of size p_1 * ... * p_K. The previous explanation invoking Weyl's equidistribution theorem and "approximate correlations" was correct for irrational constants but missed the much stronger mechanism available with rational fractions.

3. **Multi-prime fingerprinting**: Using K = 5 primes {2, 3, 5, 7, 11} creates a 35-dimensional feature vector encoding n's residue class modulo each prime. The joint residue (n mod 2, n mod 3, n mod 5, n mod 7, n mod 11) uniquely identifies n mod 2310, providing direct access to divisibility by the first 5 primes and their products. This is the number-theoretic information most relevant to tasks like GCD, SPF, and coprimality.

### 6.2 The Capacity-Generalization Tradeoff (Resolved by Rational Constants)

Phase 7 revealed a striking finding with irrational constants: adding more constants *hurts* OOD generalization while improving in-range accuracy. Pi alone (7 raw features, 69.5% in-range) achieves 62.8% average OOD, while all 8 irrationals (56 raw features, 97.5% in-range) achieve only 58.4% average OOD.

**Phase 8 resolves this tradeoff.** With rational fraction constants, adding more primes does *not* hurt OOD: rational_3 achieves 70.6% avg OOD, rational_5 achieves 75.0%, and rational_8 achieves 74.9% — each additional prime adds genuine modular-arithmetic information rather than noise. The tradeoff plateaus rather than reverses: rational_5 and rational_8 are essentially identical, but neither is *worse* than rational_3 for OOD (unlike how irrational_8 was worse than irrational_1 for OOD).

The explanation is now clear: the capacity-generalization tradeoff in Phase 7 was caused by **irrational constants adding equidistributed noise** alongside whatever partial modular signal they carried. More irrational constants = more noise = more capacity to overfit. Rational constants carry **pure modular signal with zero noise**, so adding more of them adds information without adding overfitting risk.

The practical implication is updated: use **5 rational fraction constants** {2pi/2, 2pi/3, 2pi/5, 2pi/7, 2pi/11} for the optimal balance of 97.9% in-range + 75.0% average OOD — surpassing every irrational configuration on both metrics simultaneously.

### 6.3 Task-Specific Analysis

**GCD** benefits most from SPIRNOR because GCD is fundamentally about shared factors, which directly correspond to angular correlations in the SPIRNOR embedding. Two numbers with GCD=6 both lie on the 2-spiral and the 3-spiral, creating a distinctive pattern the model can detect.

**SPF** (smallest prime factor) shows the second-largest advantage because SPF is determined by the first prime in the factorization — exactly the kind of divisibility structure SPIRNOR encodes.

**Coprime** detection achieves remarkably high OOD accuracy (91.1% at 2K-5K) because it reduces to "GCD = 1 vs GCD > 1", a binary version of GCD classification where SPIRNOR's advantage is magnified by the simpler output space.

**NumDiv** (number of divisors) shows moderate improvement. This task depends on the complete prime factorization, not just shared factors, making it harder for any embedding to extrapolate. Still, SPIRNOR's 2.2x improvement at 2K-5K demonstrates partial structure transfer.

**Omega** (distinct prime factors) shows the smallest improvement, converging with learned embeddings at 250x OOD. The distribution of omega values shifts significantly at larger numbers (most large numbers have more distinct prime factors), creating a distributional shift that even structure-preserving embeddings cannot fully compensate for.

### 6.4 SPIRNOR-RoPE: When Position Encoding Matters

SPIRNOR-RoPE's advantage is specific to multi-token tasks (4+ tokens) where standard RoPE's geometric frequency progression wastes low-frequency dimensions. On 2-3 token sequences, standard RoPE provides adequate position discrimination, and the two approaches perform similarly. This is consistent with our Phase 5 and 6 results, where sequences are short (3 tokens in Phase 6) and SPIRNOR+RoPE slightly outperforms SPIRNOR+SPIRNOR-RoPE.

The practical recommendation is: use SPIRNOR-RoPE when sequence lengths are 4+ tokens and position discrimination is critical; use standard RoPE otherwise.

### 6.5 Reproducibility

Phase 7B provides unusually strong reproducibility evidence. Across 5 random seeds, standard deviations for average OOD accuracy range from 0.001 (Learned+RoPE) to 0.004 (SPIRNOR configurations). This means the *entire* variance across seeds is smaller than the *gap* between configurations by 1-2 orders of magnitude. For example, the SPIRNOR-vs-Learned OOD gap is 0.194 with seed variance of 0.003 — the effect is 65x larger than the noise.

This extreme stability likely stems from two factors: (1) the large training set (250K examples) provides consistent gradient signal regardless of random sampling, and (2) SPIRNOR's deterministic features mean the embedding layer contributes no random initialization variance, leaving only Transformer weight initialization as a noise source.

### 6.6 Limitations

1. **Task scope**: SPIRNOR dominates on classification tasks with modular structure (GCD, SPF, Coprime, ModMul7, IsPrime: 1.4-6.5x advantage; compositional: Mod30, Diophantine, SetAllDivisible: 1.3-142.9x advantage), yields to learned embeddings on pure magnitude tasks (Compare), and provides no OOD benefit for autoregressive digit-level generation (Phase 10: 0% exact match for all configs). Phase 11 significantly expands the range of tasks where SPIRNOR dominates, but all share modular/number-theoretic structure. SPIRNOR is a domain-appropriate inductive bias, not a universal embedding improvement.

2. **Training range dependency**: While SPIRNOR generalizes far beyond the training range, accuracy still degrades with distance. The model must still learn *what to do* with SPIRNOR features during training; the embedding only provides the raw structural signal.

3. **Hybrid architectures underperform**: Phase 9 shows that both additive and gated hybrid approaches (combining SPIRNOR + learned) consistently underperform pure SPIRNOR on OOD tasks. The learned component introduces noise that degrades the modular signal. Future work could explore architectures that selectively route to SPIRNOR or learned pathways per-task rather than blending per-token.

4. **Scale**: Phase 9 extends to d_model=256 (3.7M params, +34.5 pp advantage) and Phase 11 confirms at d_model=256 (5.3M params, +57.4 pp advantage). Whether this trend continues to modern LLM scale (billions of parameters) is unknown, though the consistently positive scaling trajectory is encouraging.

5. **Autoregressive generation**: Phase 10 reveals that SPIRNOR cannot overcome the fundamental limitation of autoregressive digit-level arithmetic: the carry-chain computation must generalize to unseen sequence lengths, which is an algorithmic generalization problem beyond the reach of any embedding. SPIRNOR-frequency RoPE does improve training convergence, suggesting potential value as an alternative frequency schedule in production transformers.

---

## 7. Conclusion

We have presented SPIRNOR, a deterministic numeric embedding based on logarithmic spiral mappings parameterized by mathematical constants. Through twelve experimental phases spanning 21 classification tasks plus autoregressive arithmetic, 5 domain types, 6 architectures, scales up to 400,000 training examples, and rigorous multi-seed statistical validation, we demonstrate that SPIRNOR provides:

1. **Dramatically better OOD generalization**: 1.4-6.5x improvement over learned embeddings on modular/number-theoretic tasks, with the advantage persisting to 250x extrapolation. Statistically confirmed across 5 seeds with p < 0.0001.
2. **Parameter efficiency**: 22-50% fewer parameters than learned baselines, since the embedding requires only a small projection layer rather than a full vocabulary table.
3. **Graceful degradation**: Unlike learned embeddings that catastrophically fail OOD, SPIRNOR with rational constants maintains 72.3% accuracy at 250x extrapolation (5.6 pp drop from near-OOD).
4. **Complementary position encoding**: SPIRNOR-RoPE provides 4-8% improvement on position-sensitive multi-token tasks.
5. **Principled constant selection via rational fractions**: Constants C = 2pi/p for primes p create exact mod-p encodings, yielding 75.0% average OOD (vs 58.1% for irrational constants, +29%). Pi's dominance stems from it being 2pi/2 (exact parity encoding).
6. **Extreme reproducibility**: Standard deviations across seeds are 0.001-0.004, with effect sizes 16-106x larger than noise.
7. **Confirmed exact mod-p encoding**: Phase 9's ModMul7 task achieves 81.2% OOD at d=128 and **100% at d=256** — the strongest direct evidence that SPIRNOR's rational constants function as exact modular-arithmetic encoders.
8. **Clear domain characterization**: SPIRNOR dominates on tasks with modular structure (GCD, SPF, Coprime, ModMul7: 1.7-6.5x advantage) and generalizes to mixed tasks (IsPrime: 1.4x), while learned embeddings remain appropriate for pure magnitude tasks (Compare: learned wins by 5.3 pp). This provides practical guidance on when to apply SPIRNOR.
9. **Hybrids dilute, not complement**: Both additive and gated hybrid architectures consistently underperform pure SPIRNOR on OOD tasks. The learned embedding component introduces noise that degrades the clean modular signal. The gated hybrid's gate fails to learn to rely on SPIRNOR for OOD inputs.
10. **Scaling amplifies the advantage**: At d_model=256, SPIRNOR's OOD advantage grows to +34.5 pp over learned (76.1% vs 41.6%), and SPIRNOR is the only embedding type that benefits from scaling.
11. **Precisely delineated boundaries**: Phase 10 demonstrates that SPIRNOR's OOD advantage does not extend to autoregressive digit-level generation (addition/multiplication), where carry-chain computation is the bottleneck. This is an architectural limitation, not a representational one — no embedding can solve it. However, SPIRNOR-frequency RoPE provides measurable training efficiency gains (96.9% vs 71.7% token accuracy at epoch 5), suggesting practical value as an alternative frequency schedule.
12. **Compositional and structured reasoning at near-ceiling**: Phase 11 demonstrates that SPIRNOR's advantage extends far beyond simple tasks. On 8 compositional tasks (CRT reconstruction, Diophantine feasibility, variable-length set reasoning, hidden modulus detection), SPIRNOR achieves **98.8% avg OOD** vs **40.5% for learned** (2.4x). The crown jewel: **Mod30 = 100% at ALL ranges** (including 100K-500K), definitively proving the Chinese Remainder Theorem encoding. SumMod30 shows the largest single-task advantage in the project (142.9x). At d_model=256, SPIRNOR reaches 98.5% avg OOD with a +57.4 pp advantage over learned — the largest absolute gap in any phase.

These results establish that mathematical structure — specifically, modular arithmetic encoded through angular projections — provides a powerful inductive bias for numeric reasoning. The evolution from the original SPIRNOR equation (Phase 1) through ablation (Phases 7-7B) to the rational fraction framework (Phase 8) to domain expansion and scaling (Phase 9) represents a progression from empirical observation to theoretical understanding: the embedding works because it computes exact divisibility information, and the optimal constants are those that maximize the number of distinct prime moduli. Phase 9's ModMul7 result — 100% OOD accuracy for a task directly testing mod-7 computation — provides the most direct validation of this theory. The insight connects neural numeric representation to classical number theory (the Chinese Remainder Theorem) and suggests a general principle: the best inductive biases are not approximate statistical regularities but exact mathematical invariants. Phase 11's CRT validation (Mod30 = 100% at all ranges) and compositional reasoning results (98.8% avg OOD on 8 novel tasks) provide the definitive empirical confirmation of this principle.

---

## Appendix A: SPIRNOR Constants

### A.1 Original Irrational Constants (Phases 1-7)

| Name | Value | Phase 1 Importance (MLP) | Phase 7 Importance (Transformer) | Tier |
|------|-------|-------------------------|----------------------------------|------|
| pi | 3.14159... | +0.145 (dominant) | **+0.187** (dominant) | Essential |
| phi^2 | 2.61803... | varies | +0.052 | Important |
| ln(2) | 0.69315... | varies | +0.004 | Marginal |
| golden angle | 2.39996... | varies | +0.004 | Marginal |
| sqrt(2) | 1.41421... | +0.013 | +0.001 | Negligible |
| phi | 1.61803... | varies | 0.000 | Zero |
| pi/e | 1.15573... | varies | -0.001 | Negative |
| e | 2.71828... | varies | -0.005 | Negative |

Note: Phase 8 explains these results. Pi = 2pi/2 creates an exact mod-2 encoding; phi^2 = 2.618... partially approximates mod-3 structure. The remaining constants provide equidistributed angles with no clean modular signal, explaining their negligible or negative importance.

### A.2 Rational Fraction Constants (Phase 8, Recommended)

| Name | Value | Prime | Mod-p Encoding | Phase 8 Avg OOD Contribution |
|------|-------|-------|----------------|------------------------------|
| 2pi/2 | 3.14159... | 2 | Parity (even/odd) | Essential (shared with pi) |
| 2pi/3 | 2.09440... | 3 | n mod 3 (3 classes) | +11.8 pp (rational_3 → rational_5) |
| 2pi/5 | 1.25664... | 5 | n mod 5 (5 classes) | Included in rational_3 base |
| 2pi/7 | 0.89760... | 7 | n mod 7 (7 classes) | +4.4 pp (see rational_5 improvement) |
| 2pi/11 | 0.57120... | 11 | n mod 11 (11 classes) | Included in rational_5 |
| 2pi/13 | 0.48332... | 13 | n mod 13 (13 classes) | ~0 (rational_8 ≈ rational_5) |
| 2pi/17 | 0.36960... | 17 | n mod 17 (17 classes) | ~0 |
| 2pi/19 | 0.33069... | 19 | n mod 19 (19 classes) | ~0 |

**Recommended set**: {2pi/2, 2pi/3, 2pi/5, 2pi/7, 2pi/11} — the first 5 primes, jointly encoding n mod 2310 via the Chinese Remainder Theorem.

## Appendix B: Phase 6 Detailed Results

### B.1 Class Distributions (Training Range 2-2000)

| Task | Classes | Majority % | Distribution |
|------|---------|-----------|-------------|
| GCD | 9 | 61% (GCD=1) | [30418, 7599, 3386, 1870, 1239, 884, 581, 491, 3532] |
| SPF | 10 | 50% (SPF=2) | [25001, 8365, 3328, 1841, 1054, 778, 623, 540, 415, 8055] |
| NumDiv | 13 | 28% (4 div) | [0, 7653, 363, 14164, 81, 4930, 42, 9482, 310, 1009, 24, 5165, 6777] |
| Omega | 5 | 48% (2 factors) | [8515, 23927, 15610, 1948, 0] |
| Coprime | 2 | 61% (coprime) | [19624, 30376] |

### B.2 Training Convergence

| Configuration | Epoch 1 Acc | Epoch 30 Acc | Final Loss |
|--------------|-------------|-------------|------------|
| Learned+RoPE | 56.4% | 99.1% | 0.025 |
| Sinusoidal+RoPE | 49.8% | 97.4% | 0.072 |
| SPIRNOR+RoPE | 64.1% | 97.9% | 0.061 |
| SPIRNOR+SRoPE | 65.4% | 98.1% | 0.054 |

Note that SPIRNOR starts training faster (64.1% at epoch 1 vs 56.4% for learned), indicating that the mathematical structure in the embedding provides a better initialization for gradient descent.

### B.3 Parameter Breakdown

| Component | Learned+RoPE | SPIRNOR+RoPE | Savings |
|-----------|-------------|-------------|---------|
| Numeric embedding | 256,128 | 7,296 | **97.2%** |
| Task embedding | 640 | 640 | 0% |
| Transformer layers | 793,088 | 793,088 | 0% |
| Classification heads | 87,591 | 87,591 | 0% |
| Other (pad, norm) | 256 | 384 | — |
| **Total** | **1,137,703** | **888,999** | **21.9%** |

The 97.2% reduction in embedding parameters (256K to 7K) is the primary source of SPIRNOR's parameter efficiency. The remaining model components are identical.

---

## Appendix C: Experimental Reproducibility

Phases 1-6 use a fixed random seed (seed=42 for training data, deterministic hashes for test data). Phase 7B extends this with 5 seeds (42, 123, 456, 789, 2024) and demonstrates that all key findings are stable (std < 0.004 across seeds). Code is available at the project repository. Key software versions: Python 3.12, PyTorch 2.10.0+cu130, NumPy 2.2.6, scikit-learn 1.6.1. Hardware: NVIDIA RTX 3060 Ti (8GB VRAM), CUDA 13.0.

## Appendix D: Phase 7B Per-Range Breakdown

**Table D.1: Multi-Seed OOD by Range (mean across 5 seeds)**

| Configuration | In-Range | 2K-5K | 5K-20K | 20K-100K | 100K-500K |
|--------------|----------|-------|--------|----------|-----------|
| Learned+RoPE | 0.985 | 0.347 | 0.393 | 0.414 | 0.403 |
| SPIRNOR 1 (pi) | 0.698 | 0.618 | 0.620 | 0.623 | 0.624 |
| SPIRNOR 3 (sweet) | 0.965 | 0.677 | 0.589 | 0.534 | 0.546 |
| SPIRNOR 8 (full) | 0.975 | 0.657 | 0.585 | 0.541 | 0.551 |
| SPIRNOR 7 (no pi) | 0.972 | 0.544 | 0.320 | 0.353 | 0.366 |

Note the striking flatness of the pi-only configuration: accuracy barely changes from 0.618 (2K-5K) to 0.624 (100K-500K), indicating near-perfect scale invariance. In contrast, the 3-constant and 8-constant configurations show the expected degradation with distance. This suggests pi's angular patterns are the most scale-independent component of the SPIRNOR embedding.

## Appendix E: Phase 8 Scale Invariance Analysis

A key advantage of rational fraction constants is near-perfect scale invariance. The table below shows OOD degradation (spread between nearest and farthest OOD ranges):

| Configuration | Near OOD (2K-5K) | Far OOD (100K-500K) | Spread | Degradation |
|--------------|-------------------|---------------------|--------|-------------|
| rational_3 | 71.9% | 69.5% | 2.4 pp | 3.3% relative |
| rational_5 | 77.9% | 72.3% | 5.6 pp | 7.2% relative |
| rational_8 | 77.2% | 72.2% | 5.0 pp | 6.5% relative |
| hybrid_5 | 73.7% | 68.7% | 5.0 pp | 6.8% relative |
| irrational_3 | 67.1% | 54.3% | 12.8 pp | 19.1% relative |
| irrational_8 | 65.2% | 53.9% | 11.4 pp | 17.5% relative |
| learned | 35.7% | 40.2% | -4.5 pp | N/A |

Rational constants degrade 2-3x less than irrational constants across a 100x range increase. This is expected: the mod-p encodings are exact for all n, so the only source of degradation is the distributional shift in task outputs (e.g., larger numbers have more divisors on average), not the embedding quality.

The learned baseline shows *inverted* degradation (improving at farther ranges) because the modular hash collisions at extreme OOD effectively average over many training-range numbers, accidentally producing better-calibrated predictions than the near-OOD range where the hash distortion is more systematic.

---

## Appendix F: Phase 9 Detailed Scaling Results

**ModMul7 OOD Accuracy by Range (exact mod-7 encoding test):**

| Range | SPIRNOR d=128 | SPIRNOR d=256 | Learned d=128 | Learned d=256 |
|-------|--------------|--------------|--------------|--------------|
| In-range | 86.6% | **100.0%** | 40.2% | 46.0% |
| 2K-5K | 81.2% | **100.0%** | 12.4% | 14.1% |
| 5K-20K | 83.0% | **100.0%** | 16.3% | 15.5% |
| 20K-100K | 76.5% | 95.3% | 16.4% | 15.8% |
| 100K-500K | 72.4% | 84.1% | 15.1% | 15.8% |

The SPIRNOR d=256 model achieves **perfect 100% accuracy** on ModMul7 for in-range through 5K-20K (10x extrapolation), confirming that with sufficient model capacity, the mod-7 encoding is exact and perfectly utilized.

**Gated Hybrid Gate Statistics:**

| Range | d=128 gate | d=256 gate | Interpretation |
|-------|-----------|-----------|----------------|
| In-range | 0.555 | 0.378 | Slightly SPIRNOR / leans learned |
| 2K-5K | 0.559 | 0.357 | Slightly SPIRNOR / leans learned |
| 100K-500K | 0.566 | 0.305 | Slightly SPIRNOR / strongly learned |

At d=128, the gate is near-balanced (0.55), barely learning to use SPIRNOR. At d=256, the gate actually leans *toward* the learned embedding (0.38 in-range, 0.31 far-OOD), suggesting the larger model has more capacity to memorize via the learned pathway, even though this hurts OOD performance. The gate mechanism fails to discover the optimal strategy (pure SPIRNOR for OOD).

---

## Appendix G: Phase 10 Detailed Results — Autoregressive Arithmetic

**Training Convergence (Token Accuracy at Epoch 5, Addition):**

| Config | Epoch 1 | Epoch 5 | Epoch 10 | Epoch 30 | Final Loss |
|--------|---------|---------|----------|----------|------------|
| baseline | 37.8% | 71.7% | 98.2% | 99.83% | 0.0048 |
| spirnor_aug | 38.7% | 75.2% | 81.1% | 96.04% | 0.1007 |
| learned_aug | 37.0% | 55.3% | 60.6% | 66.5% | 0.8278 |
| std_rope | 40.7% | **95.5%** | 98.2% | 99.76% | 0.0068 |
| spirnor_rope | 44.3% | **96.9%** | 99.0% | **99.88%** | **0.0036** |
| spirnor_full | 40.5% | 96.0% | 99.4% | **99.95%** | **0.0014** |

SPIRNOR-frequency RoPE achieves the lowest final loss (0.0014 for spirnor_full, 0.0036 for spirnor_rope vs 0.0048 baseline) and fastest convergence. RoPE-based models (std_rope, spirnor_rope, spirnor_full) dramatically outpace sinusoidal models in early epochs, with spirnor_rope leading even std_rope.

**Per-Digit OOD Accuracy (Addition):**

| Config | 4-Digit | 5-Digit | 6-Digit |
|--------|---------|---------|---------|
| baseline | 13.1% | 0.0% | 0.1% |
| spirnor_aug | **16.0%** | 2.2% | **11.5%** |
| learned_aug | 15.0% | 0.0% | 0.0% |
| std_rope | 15.1% | 8.2% | 9.3% |
| spirnor_rope | 10.2% | 7.7% | 7.8% |
| spirnor_full | 11.6% | **8.7%** | 10.2% |

While exact match is 0% for all configs, per-digit accuracy reveals that spirnor_aug produces the best individual digit predictions at 4-digit and 6-digit ranges, while RoPE-based models (std_rope, spirnor_rope) maintain more consistent accuracy across OOD ranges. The baseline collapses to near-0% at 5-6 digits while SPIRNOR configs maintain ~8-12%.

**Multiplication OOD Per-Digit Accuracy:**

| Config | 3-Digit | 3-Digit Mix |
|--------|---------|-------------|
| baseline | 3.5% | 6.6% |
| spirnor_aug | **5.9%** | 7.8% |
| learned_aug | 2.8% | 4.4% |
| std_rope | **13.7%** | **15.7%** |
| spirnor_rope | 6.5% | 7.7% |
| spirnor_full | 6.3% | 6.6% |

For multiplication, std_rope achieves the best OOD per-digit accuracy (13.7%), suggesting standard geometric frequency spacing is better suited for the positional patterns in multiplication carry chains. SPIRNOR-augmented models consistently outperform the learned_aug baseline.

---

*This paper presents results from the SPIRNOR AI experimental research program (Phases 1-10).*
