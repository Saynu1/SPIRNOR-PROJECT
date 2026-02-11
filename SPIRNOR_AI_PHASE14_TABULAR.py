#!/usr/bin/env python3
"""
SPIRNOR AI Phase 14: Real-World Tabular Classification with Period-Matched Embeddings

First real-world validation of SPIRNOR. Tests whether period-matched constants
(C = 2pi/T) help generalize to unseen periodic feature values in the UCI Bike
Sharing dataset.

6-way comparison of periodic feature encoders:
  1. spirnor    — SPIRNOR with period-matched constants (our method)
  2. onehot     — Standard one-hot encoding
  3. sincos     — Standard cyclical sin/cos encoding
  4. learned    — nn.Embedding per feature
  5. linear     — Raw normalized values
  6. time2vec   — Learned sinusoidal frequencies

4 evaluation settings:
  1. random     — Standard 80/20 random split (in-distribution)
  2. hour_ood   — Train hr 0-17, test hr 18-23
  3. weekday_ood — Train weekday 0-3, test weekday 4-6
  4. month_ood  — Train month 1-8, test month 9-12

Dataset: UCI Bike Sharing (hourly), 17379 records, 4-class demand classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import json
import time
import os
import io
import zipfile
import functools
from collections import OrderedDict
from urllib.request import urlopen
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

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

# Periodic feature definitions: (name, period, column_name_in_csv)
PERIODIC_FEATURES = OrderedDict([
    ('hr',      {'period': 24, 'col': 'hr'}),
    ('weekday', {'period': 7,  'col': 'weekday'}),
    ('mnth',    {'period': 12, 'col': 'mnth'}),
])

# Non-periodic feature definitions
NUMERIC_FEATURES = ['temp', 'atemp', 'hum', 'windspeed']  # already 0-1 normalized
ONEHOT_FEATURES = {
    'weathersit': 4,  # 1-4
    'season': 4,      # 1-4
}
BINARY_FEATURES = ['holiday', 'workingday']

N_CLASSES = 4  # Quartile demand bins

# ============================================================
# DATA LOADING
# ============================================================

def download_and_load_data(data_dir=None):
    """Download UCI Bike Sharing dataset and return parsed DataFrame-like dict."""
    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(__file__))

    csv_path = os.path.join(data_dir, 'hour.csv')

    # Try local file first
    if not os.path.exists(csv_path):
        zip_path = os.path.join(data_dir, 'Bike-Sharing-Dataset.zip')
        if not os.path.exists(zip_path):
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
            print(f"Downloading UCI Bike Sharing from {url}...")
            try:
                response = urlopen(url)
                data = response.read()
                with open(zip_path, 'wb') as f:
                    f.write(data)
                print(f"  Downloaded {len(data) / 1024:.1f} KB")
            except Exception as e:
                print(f"  Download failed: {e}")
                print("  Please download manually from:")
                print(f"  {url}")
                print(f"  and place hour.csv in {data_dir}")
                raise

        # Extract hour.csv from ZIP
        print("Extracting hour.csv from ZIP...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # The CSV is inside the zip
            for name in zf.namelist():
                if name.endswith('hour.csv'):
                    with zf.open(name) as src:
                        content = src.read()
                    with open(csv_path, 'wb') as dst:
                        dst.write(content)
                    print(f"  Extracted {len(content) / 1024:.1f} KB")
                    break

    # Parse CSV manually (no pandas dependency)
    print(f"Loading {csv_path}...")
    with open(csv_path, 'r') as f:
        lines = f.readlines()

    header = lines[0].strip().split(',')
    col_idx = {name: i for i, name in enumerate(header)}

    n_rows = len(lines) - 1
    data = {}

    # Extract all needed columns
    needed_cols = (
        ['hr', 'weekday', 'mnth'] +
        NUMERIC_FEATURES +
        list(ONEHOT_FEATURES.keys()) +
        BINARY_FEATURES +
        ['cnt']
    )

    for col in needed_cols:
        vals = []
        for line in lines[1:]:
            fields = line.strip().split(',')
            vals.append(float(fields[col_idx[col]]))
        data[col] = np.array(vals)

    # Create target: 4 quartile bins of 'cnt'
    cnt = data['cnt']
    q25, q50, q75 = np.percentile(cnt, [25, 50, 75])
    labels = np.zeros(n_rows, dtype=np.int64)
    labels[cnt > q25] = 1
    labels[cnt > q50] = 2
    labels[cnt > q75] = 3
    data['labels'] = labels

    # Convert periodic features to int
    for feat in PERIODIC_FEATURES:
        data[feat] = data[PERIODIC_FEATURES[feat]['col']].astype(np.int64)

    print(f"  Loaded {n_rows} records")
    print(f"  Target distribution: {[int((labels == c).sum()) for c in range(N_CLASSES)]}")
    print(f"  Quartile thresholds: q25={q25:.0f}, q50={q50:.0f}, q75={q75:.0f}")

    return data, n_rows


# ============================================================
# NON-PERIODIC FEATURE ENCODER (shared across all configs)
# ============================================================

class NonPeriodicEncoder(nn.Module):
    """Encode non-periodic features identically for all configs.
    Returns: [batch, 14] tensor.
    """
    def __init__(self):
        super().__init__()
        # Compute output dim: numeric(4) + weathersit_oh(4) + season_oh(4) + binary(2) = 14
        self.output_dim = (
            len(NUMERIC_FEATURES) +
            sum(ONEHOT_FEATURES.values()) +
            len(BINARY_FEATURES)
        )

    def forward(self, data_batch):
        parts = []

        # Numeric features (already 0-1 normalized)
        for feat in NUMERIC_FEATURES:
            parts.append(data_batch[feat].unsqueeze(-1))

        # One-hot features
        for feat, n_cats in ONEHOT_FEATURES.items():
            idx = data_batch[feat].long() - 1  # 1-indexed -> 0-indexed
            idx = idx.clamp(0, n_cats - 1)
            oh = F.one_hot(idx, n_cats).float()
            parts.append(oh)

        # Binary features
        for feat in BINARY_FEATURES:
            parts.append(data_batch[feat].unsqueeze(-1))

        return torch.cat(parts, dim=-1)


# ============================================================
# PERIODIC FEATURE ENCODERS (6 methods)
# ============================================================

class SPIRNORPeriodicEncoder(nn.Module):
    """SPIRNOR with period-matched constants. 7 features per periodic column."""
    def __init__(self, periodic_features):
        super().__init__()
        self.feature_names = list(periodic_features.keys())
        periods = [periodic_features[f]['period'] for f in self.feature_names]
        constants = [TWO_PI / T for T in periods]
        self.register_buffer('constants', torch.tensor(constants, dtype=torch.float32))
        self.n_features = len(self.feature_names)
        self.output_dim = 7 * self.n_features

    def forward(self, data_batch):
        all_feats = []
        for i, feat_name in enumerate(self.feature_names):
            n_vals = data_batch[feat_name].float()
            C = self.constants[i]
            r = torch.log(n_vals + 1.0)  # ln(n+1) to handle n=0
            theta = (C * n_vals) % TWO_PI
            phi_angle = (PHI * n_vals) % TWO_PI
            x = r * torch.sin(theta) * torch.cos(phi_angle)
            y = r * torch.sin(theta) * torch.sin(phi_angle)
            z = r * torch.cos(theta)
            feats = torch.stack([x, y, z,
                torch.sin(theta), torch.cos(theta),
                torch.sin(phi_angle), torch.cos(phi_angle)], dim=-1)
            all_feats.append(feats)
        return torch.cat(all_feats, dim=-1)


class OneHotPeriodicEncoder(nn.Module):
    """Standard one-hot encoding for periodic features."""
    def __init__(self, periodic_features):
        super().__init__()
        self.feature_names = list(periodic_features.keys())
        self.periods = [periodic_features[f]['period'] for f in self.feature_names]
        self.output_dim = sum(self.periods)

    def forward(self, data_batch):
        parts = []
        for i, feat_name in enumerate(self.feature_names):
            idx = data_batch[feat_name].long()
            T = self.periods[i]
            idx = idx.clamp(0, T - 1)
            oh = F.one_hot(idx, T).float()
            parts.append(oh)
        return torch.cat(parts, dim=-1)


class SinCosPeriodicEncoder(nn.Module):
    """Standard cyclical encoding: sin(2pi*x/T), cos(2pi*x/T)."""
    def __init__(self, periodic_features):
        super().__init__()
        self.feature_names = list(periodic_features.keys())
        periods = [periodic_features[f]['period'] for f in self.feature_names]
        self.register_buffer('periods', torch.tensor(periods, dtype=torch.float32))
        self.output_dim = 2 * len(self.feature_names)

    def forward(self, data_batch):
        parts = []
        for i, feat_name in enumerate(self.feature_names):
            x = data_batch[feat_name].float()
            T = self.periods[i]
            angle = TWO_PI * x / T
            parts.append(torch.sin(angle).unsqueeze(-1))
            parts.append(torch.cos(angle).unsqueeze(-1))
        return torch.cat(parts, dim=-1)


class LearnedPeriodicEncoder(nn.Module):
    """nn.Embedding per periodic feature."""
    def __init__(self, periodic_features, d_embed=16):
        super().__init__()
        self.feature_names = list(periodic_features.keys())
        self.periods = [periodic_features[f]['period'] for f in self.feature_names]
        self.d_embed = d_embed
        self.embeddings = nn.ModuleDict()
        for i, feat_name in enumerate(self.feature_names):
            self.embeddings[feat_name] = nn.Embedding(self.periods[i], d_embed)
        self.output_dim = d_embed * len(self.feature_names)

    def forward(self, data_batch):
        parts = []
        for feat_name in self.feature_names:
            idx = data_batch[feat_name].long()
            T = self.embeddings[feat_name].num_embeddings
            idx = idx.clamp(0, T - 1)
            emb = self.embeddings[feat_name](idx)
            parts.append(emb)
        return torch.cat(parts, dim=-1)


class LinearPeriodicEncoder(nn.Module):
    """Simple normalized value: x/T."""
    def __init__(self, periodic_features):
        super().__init__()
        self.feature_names = list(periodic_features.keys())
        periods = [periodic_features[f]['period'] for f in self.feature_names]
        self.register_buffer('periods', torch.tensor(periods, dtype=torch.float32))
        self.output_dim = len(self.feature_names)

    def forward(self, data_batch):
        parts = []
        for i, feat_name in enumerate(self.feature_names):
            x = data_batch[feat_name].float() / self.periods[i]
            parts.append(x.unsqueeze(-1))
        return torch.cat(parts, dim=-1)


class Time2VecPeriodicEncoder(nn.Module):
    """Time2Vec: 1 linear + (d_embed-1) learned sin components per feature."""
    def __init__(self, periodic_features, d_embed=16):
        super().__init__()
        self.feature_names = list(periodic_features.keys())
        self.d_embed = d_embed
        self.n_features = len(self.feature_names)

        # Separate parameters per feature
        self.w = nn.ParameterDict()
        self.phi = nn.ParameterDict()
        for feat_name in self.feature_names:
            w = nn.Parameter(torch.empty(d_embed))
            phi = nn.Parameter(torch.empty(d_embed))
            nn.init.uniform_(w, -0.1, 0.1)
            nn.init.uniform_(phi, 0, TWO_PI)
            self.w[feat_name] = w
            self.phi[feat_name] = phi

        self.output_dim = d_embed * self.n_features

    def forward(self, data_batch):
        parts = []
        for feat_name in self.feature_names:
            x = data_batch[feat_name].float()
            raw = x.unsqueeze(-1) * self.w[feat_name] + self.phi[feat_name]
            features = torch.cat([
                raw[..., :1],                # linear component
                torch.sin(raw[..., 1:]),     # sinusoidal components
            ], dim=-1)
            parts.append(features)
        return torch.cat(parts, dim=-1)


# ============================================================
# MLP CLASSIFIER
# ============================================================

class TabularMLPClassifier(nn.Module):
    """Simple 3-layer MLP for tabular classification."""
    def __init__(self, periodic_encoder, hidden_dim=128, n_classes=N_CLASSES,
                 dropout=0.1):
        super().__init__()
        self.periodic_encoder = periodic_encoder
        self.nonperiodic_encoder = NonPeriodicEncoder()

        input_dim = periodic_encoder.output_dim + self.nonperiodic_encoder.output_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, data_batch):
        periodic_feats = self.periodic_encoder(data_batch)
        nonperiodic_feats = self.nonperiodic_encoder(data_batch)
        x = torch.cat([periodic_feats, nonperiodic_feats], dim=-1)
        return self.mlp(x)


# ============================================================
# DATA SPLITS
# ============================================================

def create_splits(data, n_rows, seed):
    """Create 4 train/test splits."""
    splits = OrderedDict()
    all_idx = np.arange(n_rows)

    # 1. Random split (80/20 stratified)
    train_idx, test_idx = train_test_split(
        all_idx, test_size=0.2, stratify=data['labels'],
        random_state=seed
    )
    splits['random'] = {'train': train_idx, 'test': test_idx}

    # 2. Hour OOD: train hr 0-17, test hr 18-23
    hr = data['hr']
    train_idx = all_idx[hr <= 17]
    test_idx = all_idx[hr >= 18]
    splits['hour_ood'] = {'train': train_idx, 'test': test_idx}

    # 3. Weekday OOD: train Mon-Thu (0-3), test Fri-Sun (4-6)
    wd = data['weekday']
    train_idx = all_idx[wd <= 3]
    test_idx = all_idx[wd >= 4]
    splits['weekday_ood'] = {'train': train_idx, 'test': test_idx}

    # 4. Month OOD: train Jan-Aug (1-8), test Sep-Dec (9-12)
    mn = data['mnth']
    train_idx = all_idx[mn <= 8]
    test_idx = all_idx[mn >= 9]
    splits['month_ood'] = {'train': train_idx, 'test': test_idx}

    return splits


def make_batch(data, indices, dev):
    """Extract a batch dict from data given indices."""
    batch = {}
    for key in data:
        arr = data[key][indices]
        if key in ['labels'] or key in PERIODIC_FEATURES or key in ONEHOT_FEATURES or key in ['holiday', 'workingday']:
            batch[key] = torch.tensor(arr, dtype=torch.long, device=dev)
        else:
            batch[key] = torch.tensor(arr, dtype=torch.float32, device=dev)
    return batch


# ============================================================
# TRAINING AND EVALUATION
# ============================================================

def train_model(model, data, train_idx, epochs=100, batch_size=256, lr=1e-3):
    """Train model on subset of data."""
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    n = len(train_idx)

    for epoch in range(epochs):
        perm = np.random.permutation(n)
        total_loss = 0.0
        total_correct = 0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = train_idx[perm[i:i + batch_size]]
            batch = make_batch(data, idx, device)

            optimizer.zero_grad()
            logits = model(batch)
            loss = F.cross_entropy(logits, batch['labels'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_correct += (logits.argmax(dim=-1) == batch['labels']).sum().item()
            n_batches += 1

        if (epoch + 1) % 50 == 0 or epoch == 0:
            avg_loss = total_loss / n_batches
            avg_acc = total_correct / n
            print(f"    Epoch {epoch+1:3d}: loss={avg_loss:.4f}, acc={avg_acc:.3f}")


@torch.no_grad()
def evaluate_model(model, data, test_idx, batch_size=2048):
    """Evaluate model, return accuracy and macro F1."""
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    for i in range(0, len(test_idx), batch_size):
        idx = test_idx[i:i + batch_size]
        batch = make_batch(data, idx, device)
        logits = model(batch)
        preds = logits.argmax(dim=-1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(batch['labels'].cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    accuracy = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return {'accuracy': round(float(accuracy), 4),
            'macro_f1': round(float(macro_f1), 4)}


# ============================================================
# ENCODING CONFIGS
# ============================================================

ENCODING_CONFIGS = OrderedDict([
    ('spirnor', {
        'class': SPIRNORPeriodicEncoder,
        'kwargs': {'periodic_features': PERIODIC_FEATURES},
        'desc': 'SPIRNOR period-matched: C=2pi/T per feature',
    }),
    ('onehot', {
        'class': OneHotPeriodicEncoder,
        'kwargs': {'periodic_features': PERIODIC_FEATURES},
        'desc': 'One-hot encoding per periodic feature',
    }),
    ('sincos', {
        'class': SinCosPeriodicEncoder,
        'kwargs': {'periodic_features': PERIODIC_FEATURES},
        'desc': 'Standard cyclical: sin(2pi*x/T), cos(2pi*x/T)',
    }),
    ('learned', {
        'class': LearnedPeriodicEncoder,
        'kwargs': {'periodic_features': PERIODIC_FEATURES, 'd_embed': 16},
        'desc': 'nn.Embedding(T, 16) per feature',
    }),
    ('linear', {
        'class': LinearPeriodicEncoder,
        'kwargs': {'periodic_features': PERIODIC_FEATURES},
        'desc': 'Normalized: x/T per feature',
    }),
    ('time2vec', {
        'class': Time2VecPeriodicEncoder,
        'kwargs': {'periodic_features': PERIODIC_FEATURES, 'd_embed': 16},
        'desc': 'Time2Vec: 1 linear + 15 learned sin per feature',
    }),
])

SPLIT_NAMES = ['random', 'hour_ood', 'weekday_ood', 'month_ood']
SEEDS = [42, 43, 44, 45, 46]


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_experiment():
    """Run full Phase 14 protocol: 6 configs x 4 splits x 5 seeds = 120 runs."""

    print("=" * 70)
    print("SPIRNOR AI Phase 14: Real-World Tabular Classification")
    print("=" * 70)

    # Load data
    data, n_rows = download_and_load_data()

    # Print split sizes
    sample_splits = create_splits(data, n_rows, seed=42)
    print("\nSplit sizes:")
    for split_name, split_idx in sample_splits.items():
        n_train = len(split_idx['train'])
        n_test = len(split_idx['test'])
        print(f"  {split_name:15s}: train={n_train:6d}, test={n_test:5d}")

    # Print encoder info
    print("\nEncoding configurations:")
    for name, cfg in ENCODING_CONFIGS.items():
        enc = cfg['class'](**cfg['kwargs'])
        print(f"  {name:10s}: periodic_dim={enc.output_dim:3d}, desc={cfg['desc']}")

    # Results storage
    results = {config: {split: [] for split in SPLIT_NAMES}
               for config in ENCODING_CONFIGS}

    t_start = time.time()
    total_runs = len(ENCODING_CONFIGS) * len(SPLIT_NAMES) * len(SEEDS)
    run_count = 0

    for config_name, config_def in ENCODING_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Config: {config_name} — {config_def['desc']}")
        print(f"{'='*60}")

        for split_name in SPLIT_NAMES:
            print(f"\n  Split: {split_name}")

            for seed in SEEDS:
                run_count += 1
                torch.manual_seed(seed)
                np.random.seed(seed)

                # Create encoder and model
                encoder = config_def['class'](**config_def['kwargs'])
                model = TabularMLPClassifier(encoder)
                n_params = sum(p.numel() for p in model.parameters())

                # Create splits
                splits = create_splits(data, n_rows, seed)
                train_idx = splits[split_name]['train']
                test_idx = splits[split_name]['test']

                # Train
                print(f"    Seed {seed} ({run_count}/{total_runs}, params={n_params}):")
                train_model(model, data, train_idx, epochs=100, batch_size=256)

                # Evaluate
                metrics = evaluate_model(model, data, test_idx)

                result = {
                    'accuracy': metrics['accuracy'],
                    'macro_f1': metrics['macro_f1'],
                    'n_params': n_params,
                    'seed': seed,
                }
                results[config_name][split_name].append(result)

                print(f"    -> acc={metrics['accuracy']:.4f}, f1={metrics['macro_f1']:.4f}")

    total_time = time.time() - t_start
    print(f"\n\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # ============================================================
    # AGGREGATE AND PRINT SUMMARY
    # ============================================================

    print("\n" + "=" * 70)
    print("PHASE 14 RESULTS SUMMARY")
    print("=" * 70)

    summary = {}
    for config_name in ENCODING_CONFIGS:
        summary[config_name] = {}
        for split_name in SPLIT_NAMES:
            runs = results[config_name][split_name]
            accs = [r['accuracy'] for r in runs]
            f1s = [r['macro_f1'] for r in runs]
            summary[config_name][split_name] = {
                'accuracy_mean': round(float(np.mean(accs)), 4),
                'accuracy_std': round(float(np.std(accs)), 4),
                'f1_mean': round(float(np.mean(f1s)), 4),
                'f1_std': round(float(np.std(f1s)), 4),
                'n_params': runs[0]['n_params'],
            }

    # Print table per split
    for split_name in SPLIT_NAMES:
        print(f"\n--- {split_name.upper()} ---")
        print(f"{'Config':12s} | {'Accuracy':18s} | {'Macro F1':18s} | {'Params':>8s}")
        print("-" * 65)
        for config_name in ENCODING_CONFIGS:
            s = summary[config_name][split_name]
            acc_str = f"{s['accuracy_mean']:.4f} +/- {s['accuracy_std']:.4f}"
            f1_str = f"{s['f1_mean']:.4f} +/- {s['f1_std']:.4f}"
            print(f"{config_name:12s} | {acc_str:18s} | {f1_str:18s} | {s['n_params']:>8d}")

    # Print OOD summary (average across 3 OOD splits)
    print(f"\n--- AVERAGE OOD (across hour_ood, weekday_ood, month_ood) ---")
    print(f"{'Config':12s} | {'Avg OOD Acc':12s} | {'Avg OOD F1':12s} | {'Ratio vs Random':16s}")
    print("-" * 60)
    for config_name in ENCODING_CONFIGS:
        ood_accs = []
        for split in ['hour_ood', 'weekday_ood', 'month_ood']:
            ood_accs.append(summary[config_name][split]['accuracy_mean'])
        avg_ood = np.mean(ood_accs)
        random_acc = summary[config_name]['random']['accuracy_mean']
        ratio = avg_ood / random_acc if random_acc > 0 else 0
        print(f"{config_name:12s} | {avg_ood:12.4f} | {np.mean([summary[config_name][s]['f1_mean'] for s in ['hour_ood', 'weekday_ood', 'month_ood']]):12.4f} | {ratio:16.3f}")

    # Statistical tests: SPIRNOR vs each baseline on OOD splits
    print(f"\n--- STATISTICAL TESTS (SPIRNOR vs baselines, paired t-test on OOD accuracy) ---")
    from scipy import stats as scipy_stats

    for split_name in ['hour_ood', 'weekday_ood', 'month_ood']:
        print(f"\n  {split_name}:")
        spirnor_accs = [r['accuracy'] for r in results['spirnor'][split_name]]
        for config_name in ENCODING_CONFIGS:
            if config_name == 'spirnor':
                continue
            other_accs = [r['accuracy'] for r in results[config_name][split_name]]
            diff = np.array(spirnor_accs) - np.array(other_accs)
            if np.std(diff) > 0:
                t_stat, p_val = scipy_stats.ttest_rel(spirnor_accs, other_accs)
            else:
                t_stat, p_val = float('inf'), 0.0
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
            print(f"    vs {config_name:10s}: diff={np.mean(diff):+.4f}, t={t_stat:+6.2f}, p={p_val:.4f} {sig}")

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    output = {
        'phase': 'Phase 14: Real-World Tabular Classification',
        'dataset': 'UCI Bike Sharing (hourly)',
        'n_rows': n_rows,
        'n_classes': N_CLASSES,
        'periodic_features': {k: v['period'] for k, v in PERIODIC_FEATURES.items()},
        'configs': list(ENCODING_CONFIGS.keys()),
        'splits': SPLIT_NAMES,
        'seeds': SEEDS,
        'total_time': round(total_time, 1),
        'summary': summary,
        'raw_results': {
            config: {
                split: results[config][split]
                for split in SPLIT_NAMES
            }
            for config in ENCODING_CONFIGS
        },
    }

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, 'SPIRNOR_AI_PHASE14_RESULTS.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return summary


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    run_experiment()
