#!/usr/bin/env python3
"""
SPIRNOR AI Phase 14B: Cross-Domain Tabular Validation (Two-Tier Framework)

Validates the two-tier encoding framework from Phase 14 across 3 additional
real-world datasets:
  1. Seoul Bike Sharing (UCI #560) — hourly bike demand, 2017-2018
  2. Metro Interstate Traffic (UCI #492) — hourly traffic volume, 2012-2018
  3. Beijing PM2.5 (UCI #381) — hourly air quality, 2010-2014

Same 6 encoding methods as Phase 14:
  spirnor, onehot, sincos, learned, linear, time2vec

Same evaluation protocol: random + OOD splits, 5 seeds, accuracy + macro F1.
Cross-dataset analysis: Friedman test + average ranking.
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
import csv
import zipfile
import functools
from collections import OrderedDict
from urllib.request import urlopen, Request
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

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

SEEDS = [42, 43, 44, 45, 46]

# ============================================================
# DATASET CONFIGURATIONS
# ============================================================

DATASET_CONFIGS = OrderedDict([
    ('seoul_bike', {
        'name': 'Seoul Bike Sharing',
        'periodic_features': OrderedDict([
            ('hour', {'period': 24}),
            ('month', {'period': 12}),
        ]),
        'numeric_features': [
            'temperature', 'humidity', 'wind_speed', 'visibility',
            'dew_point', 'solar_radiation', 'rainfall', 'snowfall',
        ],
        'categorical_features': OrderedDict([
            ('seasons', 4),
        ]),
        'binary_features': ['holiday', 'functioning_day'],
        'n_classes': 4,
        'splits': OrderedDict([
            ('random', {'type': 'random', 'test_size': 0.2}),
            ('hour_ood', {'type': 'ood', 'feature': 'hour', 'train_max': 17, 'test_min': 18}),
            ('month_ood', {'type': 'ood', 'feature': 'month', 'train_max': 8, 'test_min': 9}),
        ]),
    }),
    ('metro_traffic', {
        'name': 'Metro Interstate Traffic',
        'periodic_features': OrderedDict([
            ('hour', {'period': 24}),
            ('day_of_week', {'period': 7}),
            ('month', {'period': 12}),
        ]),
        'numeric_features': ['temp', 'rain_1h', 'snow_1h', 'clouds_all'],
        'categorical_features': OrderedDict([
            ('weather_main', 11),
        ]),
        'binary_features': ['holiday'],
        'n_classes': 4,
        'splits': OrderedDict([
            ('random', {'type': 'random', 'test_size': 0.2}),
            ('hour_ood', {'type': 'ood', 'feature': 'hour', 'train_max': 17, 'test_min': 18}),
            ('weekday_ood', {'type': 'ood', 'feature': 'day_of_week', 'train_max': 3, 'test_min': 4}),
            ('month_ood', {'type': 'ood', 'feature': 'month', 'train_max': 8, 'test_min': 9}),
        ]),
    }),
    ('beijing_pm25', {
        'name': 'Beijing PM2.5',
        'periodic_features': OrderedDict([
            ('hour', {'period': 24}),
            ('month', {'period': 12}),
        ]),
        'numeric_features': ['DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir'],
        'categorical_features': OrderedDict([
            ('cbwd', 4),
        ]),
        'binary_features': [],
        'n_classes': 4,
        'splits': OrderedDict([
            ('random', {'type': 'random', 'test_size': 0.2}),
            ('hour_ood', {'type': 'ood', 'feature': 'hour', 'train_max': 17, 'test_min': 18}),
            ('month_ood', {'type': 'ood', 'feature': 'month', 'train_max': 8, 'test_min': 9}),
        ]),
    }),
])


# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================

def _download_file(url, dest_path):
    """Download a file with a browser-like User-Agent."""
    print(f"  Downloading from {url}...")
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    response = urlopen(req, timeout=60)
    raw = response.read()
    with open(dest_path, 'wb') as f:
        f.write(raw)
    print(f"  Downloaded {len(raw) / 1024:.1f} KB")
    return raw


def _parse_csv_lines(lines, needed_cols=None):
    """Parse CSV lines into dict of numpy arrays."""
    reader = csv.reader(lines)
    header = next(reader)
    col_idx = {name.strip(): i for i, name in enumerate(header)}

    if needed_cols is None:
        needed_cols = list(col_idx.keys())

    rows = []
    for row in reader:
        if len(row) < len(header):
            continue
        rows.append(row)

    data = {}
    for col in needed_cols:
        if col not in col_idx:
            continue
        idx = col_idx[col]
        vals = []
        for row in rows:
            try:
                vals.append(float(row[idx]))
            except (ValueError, IndexError):
                vals.append(float('nan'))
        data[col] = np.array(vals)

    return data, len(rows)


def load_seoul_bike(data_dir):
    """Load Seoul Bike Sharing dataset."""
    csv_path = os.path.join(data_dir, 'SeoulBikeData.csv')

    if not os.path.exists(csv_path):
        zip_path = os.path.join(data_dir, 'seoul_bike.zip')
        if not os.path.exists(zip_path):
            url = "https://archive.ics.uci.edu/static/public/560/seoul+bike+sharing+demand.zip"
            _download_file(url, zip_path)

        print("  Extracting Seoul Bike data...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                if name.endswith('.csv'):
                    with zf.open(name) as src:
                        content = src.read()
                    with open(csv_path, 'wb') as dst:
                        dst.write(content)
                    break

    # Parse CSV (Seoul uses mixed encoding — try several)
    for enc in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'euc-kr']:
        try:
            with open(csv_path, 'r', encoding=enc) as f:
                lines = list(f)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue

    reader = csv.reader(lines)
    header = next(reader)
    header = [h.strip() for h in header]
    col_idx = {name: i for i, name in enumerate(header)}

    rows = [row for row in reader if len(row) >= len(header)]
    n_rows = len(rows)

    # Map column names (dataset has specific names)
    # Columns: Date, Rented Bike Count, Hour, Temperature, Humidity, Wind speed,
    # Visibility, Dew point temperature, Solar Radiation, Rainfall, Snowfall,
    # Seasons, Holiday, Functioning Day
    # Normalize header: strip degree symbols and extra spaces for matching
    def _norm(s):
        # Remove degree symbols (various encodings), normalize spaces
        import re
        s = re.sub(r'[°\u00b0\xb0]', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    norm_header = [_norm(h) for h in header]
    norm_col_idx = {name: i for i, name in enumerate(norm_header)}

    col_map = {
        'Hour': 'hour',
        'Rented Bike Count': 'target',
        'Temperature(C)': 'temperature',
        'Humidity(%)': 'humidity',
        'Wind speed (m/s)': 'wind_speed',
        'Wind speed(m/s)': 'wind_speed',
        'Visibility (10m)': 'visibility',
        'Visibility(10m)': 'visibility',
        'Dew point temperature(C)': 'dew_point',
        'Solar Radiation (MJ/m2)': 'solar_radiation',
        'Solar Radiation(MJ/m2)': 'solar_radiation',
        'Rainfall(mm)': 'rainfall',
        'Snowfall (cm)': 'snowfall',
        'Snowfall(cm)': 'snowfall',
        'Seasons': 'seasons',
        'Holiday': 'holiday',
        'Functioning Day': 'functioning_day',
    }

    # Also try to find Month from Date column
    date_idx = None
    for possible in ['Date', 'date']:
        if possible in col_idx:
            date_idx = col_idx[possible]
            break
        if possible in norm_col_idx:
            date_idx = norm_col_idx[possible]
            break

    data = {}

    # Extract columns using flexible mapping (try both raw and normalized headers)
    for csv_col, our_name in col_map.items():
        # Try raw header first, then normalized
        if csv_col in col_idx:
            idx = col_idx[csv_col]
        elif csv_col in norm_col_idx:
            idx = norm_col_idx[csv_col]
        else:
            continue

        vals = []
        for row in rows:
            v = row[idx].strip()
            if our_name == 'holiday':
                vals.append(1.0 if v in ['Holiday', '1', 'Yes'] else 0.0)
            elif our_name == 'functioning_day':
                vals.append(1.0 if v in ['Yes', '1'] else 0.0)
            elif our_name == 'seasons':
                season_map = {'Spring': 0, 'Summer': 1, 'Autumn': 2, 'Fall': 2, 'Winter': 3}
                if v in season_map:
                    vals.append(float(season_map[v]))
                else:
                    try:
                        vals.append(float(v))
                    except ValueError:
                        vals.append(float('nan'))
            else:
                try:
                    vals.append(float(v))
                except ValueError:
                    vals.append(float('nan'))
        data[our_name] = np.array(vals)

    # Extract month from Date (format: DD/MM/YYYY or similar)
    if date_idx is not None:
        months = []
        for row in rows:
            date_str = row[date_idx].strip()
            try:
                # Try DD/MM/YYYY format
                parts = date_str.replace('-', '/').split('/')
                if len(parts) >= 3:
                    # Could be DD/MM/YYYY or MM/DD/YYYY
                    # Seoul data uses DD/MM/YYYY
                    month = int(parts[1])
                    if month > 12:  # Actually MM/DD/YYYY
                        month = int(parts[0])
                    months.append(float(month))
                else:
                    months.append(float('nan'))
            except (ValueError, IndexError):
                months.append(float('nan'))
        data['month'] = np.array(months)

    # Handle seasons: map to 1-4 if string
    if 'seasons' in data:
        s = data['seasons']
        # If it's already numeric 1-4, keep it
        # Some versions have: Winter=1, Spring=2, Summer=3, Autumn=4
        data['seasons'] = s.astype(np.float64)

    # Drop NaN rows (only check float arrays)
    valid_mask = np.ones(n_rows, dtype=bool)
    for key in data:
        arr = data[key]
        if arr.dtype in [np.float64, np.float32]:
            valid_mask &= ~np.isnan(arr)
    n_valid = int(valid_mask.sum())

    for key in data:
        data[key] = data[key][valid_mask]

    # Convert periodic features to int
    data['hour'] = data['hour'].astype(np.int64)
    data['month'] = data['month'].astype(np.int64)

    # Create quartile target
    target = data.pop('target')
    q25, q50, q75 = np.percentile(target, [25, 50, 75])
    labels = np.zeros(n_valid, dtype=np.int64)
    labels[target > q25] = 1
    labels[target > q50] = 2
    labels[target > q75] = 3
    data['labels'] = labels

    print(f"  Seoul Bike: {n_valid} records (from {n_rows}), "
          f"target dist: {[int((labels == c).sum()) for c in range(4)]}")

    return data, n_valid


def load_metro_traffic(data_dir):
    """Load Metro Interstate Traffic dataset."""
    csv_path = os.path.join(data_dir, 'Metro_Interstate_Traffic_Volume.csv')

    if not os.path.exists(csv_path):
        zip_path = os.path.join(data_dir, 'metro_traffic.zip')
        if not os.path.exists(zip_path):
            url = "https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip"
            _download_file(url, zip_path)

        import gzip as _gzip
        print("  Extracting Metro Traffic data...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                if '.csv' in name:
                    with zf.open(name) as src:
                        content = src.read()
                    # Handle .csv.gz (double compression)
                    if name.endswith('.gz'):
                        content = _gzip.decompress(content)
                    with open(csv_path, 'wb') as dst:
                        dst.write(content)
                    break

    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = list(f)

    reader = csv.reader(lines)
    header = next(reader)
    header = [h.strip() for h in header]
    col_idx = {name: i for i, name in enumerate(header)}

    rows = [row for row in reader if len(row) >= len(header)]
    n_rows = len(rows)

    # Columns: holiday, temp, rain_1h, snow_1h, clouds_all, weather_main,
    #          weather_description, date_time, traffic_volume
    data = {}

    # Numeric features
    for feat in ['temp', 'rain_1h', 'snow_1h', 'clouds_all']:
        idx = col_idx[feat]
        vals = []
        for row in rows:
            try:
                vals.append(float(row[idx].strip()))
            except ValueError:
                vals.append(float('nan'))
        data[feat] = np.array(vals)

    # Target: traffic_volume
    target_idx = col_idx['traffic_volume']
    target = np.array([float(row[target_idx].strip()) for row in rows])

    # Holiday: binary (is it a holiday name or "None")
    holiday_idx = col_idx['holiday']
    data['holiday'] = np.array([
        0.0 if row[holiday_idx].strip() == 'None' else 1.0
        for row in rows
    ])

    # Weather main: categorical -> integer encoding
    weather_idx = col_idx['weather_main']
    weather_vals = [row[weather_idx].strip() for row in rows]
    weather_cats = sorted(set(weather_vals))
    weather_map = {cat: i for i, cat in enumerate(weather_cats)}
    data['weather_main'] = np.array([float(weather_map[w]) for w in weather_vals])

    # Extract hour, day_of_week, month from date_time
    dt_idx = col_idx['date_time']
    hours = []
    dows = []
    months = []
    for row in rows:
        dt_str = row[dt_idx].strip()
        try:
            # Format: "2012-10-02 09:00:00"
            date_part, time_part = dt_str.split(' ')
            y, m, d = date_part.split('-')
            h = time_part.split(':')[0]
            hours.append(int(h))
            months.append(int(m))
            # Compute day of week using Zeller-like or Python's approach
            import datetime
            dt = datetime.date(int(y), int(m), int(d))
            dows.append(dt.weekday())  # Monday=0, Sunday=6
        except (ValueError, IndexError):
            hours.append(-1)
            months.append(-1)
            dows.append(-1)

    data['hour'] = np.array(hours, dtype=np.int64)
    data['day_of_week'] = np.array(dows, dtype=np.int64)
    data['month'] = np.array(months, dtype=np.int64)

    # Drop duplicates and invalid rows
    valid_mask = np.ones(n_rows, dtype=bool)
    for key in data:
        if data[key].dtype == np.float64:
            valid_mask &= ~np.isnan(data[key])
    valid_mask &= (data['hour'] >= 0)

    # Remove duplicate timestamps (take first occurrence)
    # Create a unique key per timestamp
    seen = set()
    unique_mask = np.zeros(n_rows, dtype=bool)
    for i in range(n_rows):
        if valid_mask[i]:
            key = (int(data['hour'][i]), int(data['day_of_week'][i]),
                   int(data['month'][i]), rows[i][dt_idx].strip())
            if key not in seen:
                seen.add(key)
                unique_mask[i] = True

    for key in data:
        data[key] = data[key][unique_mask]
    target = target[unique_mask]
    n_valid = unique_mask.sum()

    # Update weather_main cardinality
    actual_cats = len(set(data['weather_main'].astype(int)))

    # Create quartile target
    q25, q50, q75 = np.percentile(target, [25, 50, 75])
    labels = np.zeros(n_valid, dtype=np.int64)
    labels[target > q25] = 1
    labels[target > q50] = 2
    labels[target > q75] = 3
    data['labels'] = labels

    print(f"  Metro Traffic: {n_valid} records (from {n_rows}), "
          f"weather categories: {actual_cats}, "
          f"target dist: {[int((labels == c).sum()) for c in range(4)]}")

    return data, n_valid


def load_beijing_pm25(data_dir):
    """Load Beijing PM2.5 dataset."""
    csv_path = os.path.join(data_dir, 'PRSA_data_2010.1.1-2014.12.31.csv')

    if not os.path.exists(csv_path):
        zip_path = os.path.join(data_dir, 'beijing_pm25.zip')
        if not os.path.exists(zip_path):
            url = "https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip"
            _download_file(url, zip_path)

        print("  Extracting Beijing PM2.5 data...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                if name.endswith('.csv'):
                    with zf.open(name) as src:
                        content = src.read()
                    with open(csv_path, 'wb') as dst:
                        dst.write(content)
                    break

    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = list(f)

    reader = csv.reader(lines)
    header = next(reader)
    header = [h.strip() for h in header]
    col_idx = {name: i for i, name in enumerate(header)}

    rows = [row for row in reader if len(row) >= len(header)]
    n_rows = len(rows)

    # Columns: No, year, month, day, hour, pm2.5, DEWP, TEMP, PRES, cbwd, Iws, Is, Ir
    data = {}

    # Periodic features
    for feat in ['hour', 'month']:
        idx = col_idx[feat]
        data[feat] = np.array([int(row[idx].strip()) for row in rows], dtype=np.int64)

    # Numeric features
    for feat in ['DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']:
        idx = col_idx[feat]
        vals = []
        for row in rows:
            try:
                vals.append(float(row[idx].strip()))
            except ValueError:
                vals.append(float('nan'))
        data[feat] = np.array(vals)

    # Target: pm2.5
    pm25_idx = col_idx['pm2.5']
    target = []
    for row in rows:
        try:
            target.append(float(row[pm25_idx].strip()))
        except ValueError:
            target.append(float('nan'))
    target = np.array(target)

    # Wind direction: categorical
    cbwd_idx = col_idx['cbwd']
    cbwd_vals = [row[cbwd_idx].strip() for row in rows]
    cbwd_cats = sorted(set(v for v in cbwd_vals if v))
    cbwd_map = {cat: i for i, cat in enumerate(cbwd_cats)}
    data['cbwd'] = np.array([float(cbwd_map.get(v, 0)) for v in cbwd_vals])

    # Drop NaN rows (pm2.5 has many NaNs)
    valid_mask = ~np.isnan(target)
    for key in data:
        if data[key].dtype == np.float64:
            valid_mask &= ~np.isnan(data[key])

    for key in data:
        data[key] = data[key][valid_mask]
    target = target[valid_mask]
    n_valid = int(valid_mask.sum())

    # Create quartile target
    q25, q50, q75 = np.percentile(target, [25, 50, 75])
    labels = np.zeros(n_valid, dtype=np.int64)
    labels[target > q25] = 1
    labels[target > q50] = 2
    labels[target > q75] = 3
    data['labels'] = labels

    print(f"  Beijing PM2.5: {n_valid} records (from {n_rows}), "
          f"wind dirs: {len(cbwd_cats)}, "
          f"target dist: {[int((labels == c).sum()) for c in range(4)]}")

    return data, n_valid


DATASET_LOADERS = {
    'seoul_bike': load_seoul_bike,
    'metro_traffic': load_metro_traffic,
    'beijing_pm25': load_beijing_pm25,
}


# ============================================================
# GENERALIZED NON-PERIODIC ENCODER
# ============================================================

class NonPeriodicEncoder(nn.Module):
    """Encode non-periodic features. Parameterized per dataset."""
    def __init__(self, numeric_features, categorical_features, binary_features):
        super().__init__()
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features  # OrderedDict: name -> n_cats
        self.binary_features = binary_features
        self.output_dim = (
            len(numeric_features) +
            sum(categorical_features.values()) +
            len(binary_features)
        )

    def forward(self, data_batch):
        parts = []

        for feat in self.numeric_features:
            parts.append(data_batch[feat].unsqueeze(-1))

        for feat, n_cats in self.categorical_features.items():
            idx = data_batch[feat].long()
            idx = idx.clamp(0, n_cats - 1)
            oh = F.one_hot(idx, n_cats).float()
            parts.append(oh)

        for feat in self.binary_features:
            parts.append(data_batch[feat].unsqueeze(-1))

        if len(parts) == 0:
            # No non-periodic features (unlikely but safe)
            batch_size = next(iter(data_batch.values())).shape[0]
            return torch.zeros(batch_size, 0, device=next(iter(data_batch.values())).device)

        return torch.cat(parts, dim=-1)


# ============================================================
# PERIODIC FEATURE ENCODERS (same 6 as Phase 14)
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
            r = torch.log(n_vals + 1.0)
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
        self.w = nn.ParameterDict()
        self.phi = nn.ParameterDict()
        for feat_name in self.feature_names:
            w = nn.Parameter(torch.empty(d_embed))
            phi_p = nn.Parameter(torch.empty(d_embed))
            nn.init.uniform_(w, -0.1, 0.1)
            nn.init.uniform_(phi_p, 0, TWO_PI)
            self.w[feat_name] = w
            self.phi[feat_name] = phi_p
        self.output_dim = d_embed * self.n_features

    def forward(self, data_batch):
        parts = []
        for feat_name in self.feature_names:
            x = data_batch[feat_name].float()
            raw = x.unsqueeze(-1) * self.w[feat_name] + self.phi[feat_name]
            features = torch.cat([
                raw[..., :1],
                torch.sin(raw[..., 1:]),
            ], dim=-1)
            parts.append(features)
        return torch.cat(parts, dim=-1)


# ============================================================
# MLP CLASSIFIER (generalized)
# ============================================================

class TabularMLPClassifier(nn.Module):
    """Simple 3-layer MLP for tabular classification."""
    def __init__(self, periodic_encoder, nonperiodic_encoder,
                 hidden_dim=128, n_classes=4, dropout=0.1):
        super().__init__()
        self.periodic_encoder = periodic_encoder
        self.nonperiodic_encoder = nonperiodic_encoder
        input_dim = periodic_encoder.output_dim + nonperiodic_encoder.output_dim
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
# DATA SPLITS (generalized)
# ============================================================

def create_splits(data, n_rows, split_configs, seed):
    """Create train/test splits from config."""
    splits = OrderedDict()
    all_idx = np.arange(n_rows)

    for split_name, cfg in split_configs.items():
        if cfg['type'] == 'random':
            train_idx, test_idx = train_test_split(
                all_idx, test_size=cfg['test_size'],
                stratify=data['labels'], random_state=seed
            )
        elif cfg['type'] == 'ood':
            feat_vals = data[cfg['feature']]
            train_idx = all_idx[feat_vals <= cfg['train_max']]
            test_idx = all_idx[feat_vals >= cfg['test_min']]
        else:
            raise ValueError(f"Unknown split type: {cfg['type']}")

        splits[split_name] = {'train': train_idx, 'test': test_idx}

    return splits


# ============================================================
# BATCH CREATION WITH NORMALIZATION
# ============================================================

def normalize_numeric_features(data, train_idx, numeric_features):
    """Fit StandardScaler on train, return scaler and normalized data copy."""
    if not numeric_features:
        return data, None

    # Build train matrix
    train_matrix = np.column_stack([data[f][train_idx] for f in numeric_features])
    scaler = StandardScaler()
    scaler.fit(train_matrix)

    # Normalize ALL data (but scaler was fit only on train)
    all_matrix = np.column_stack([data[f] for f in numeric_features])
    normalized = scaler.transform(all_matrix)

    # Create copy with normalized values
    data_norm = dict(data)
    for i, feat in enumerate(numeric_features):
        data_norm[feat] = normalized[:, i]

    return data_norm, scaler


def make_batch(data, indices, dataset_cfg, dev):
    """Extract a batch dict from data given indices."""
    batch = {}
    periodic_names = set(dataset_cfg['periodic_features'].keys())
    categorical_names = set(dataset_cfg['categorical_features'].keys())
    binary_names = set(dataset_cfg['binary_features'])

    for key in data:
        arr = data[key][indices]
        if key == 'labels' or key in periodic_names or key in categorical_names:
            batch[key] = torch.tensor(arr, dtype=torch.long, device=dev)
        elif key in binary_names:
            batch[key] = torch.tensor(arr, dtype=torch.float32, device=dev)
        else:
            batch[key] = torch.tensor(arr, dtype=torch.float32, device=dev)

    return batch


# ============================================================
# TRAINING AND EVALUATION
# ============================================================

def train_model(model, data, train_idx, dataset_cfg, epochs=100, batch_size=256, lr=1e-3):
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
            batch = make_batch(data, idx, dataset_cfg, device)

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
            print(f"      Epoch {epoch+1:3d}: loss={avg_loss:.4f}, acc={avg_acc:.3f}")


@torch.no_grad()
def evaluate_model(model, data, test_idx, dataset_cfg, batch_size=2048):
    """Evaluate model, return accuracy and macro F1."""
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    for i in range(0, len(test_idx), batch_size):
        idx = test_idx[i:i + batch_size]
        batch = make_batch(data, idx, dataset_cfg, device)
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
# ENCODING CONFIGS (built per dataset)
# ============================================================

def build_encoding_configs(periodic_features):
    """Build the 6 encoding configs for a given set of periodic features."""
    return OrderedDict([
        ('spirnor', {
            'class': SPIRNORPeriodicEncoder,
            'kwargs': {'periodic_features': periodic_features},
            'desc': 'SPIRNOR period-matched: C=2pi/T',
        }),
        ('onehot', {
            'class': OneHotPeriodicEncoder,
            'kwargs': {'periodic_features': periodic_features},
            'desc': 'One-hot encoding',
        }),
        ('sincos', {
            'class': SinCosPeriodicEncoder,
            'kwargs': {'periodic_features': periodic_features},
            'desc': 'Cyclical: sin/cos(2pi*x/T)',
        }),
        ('learned', {
            'class': LearnedPeriodicEncoder,
            'kwargs': {'periodic_features': periodic_features, 'd_embed': 16},
            'desc': 'nn.Embedding(T, 16)',
        }),
        ('linear', {
            'class': LinearPeriodicEncoder,
            'kwargs': {'periodic_features': periodic_features},
            'desc': 'Normalized: x/T',
        }),
        ('time2vec', {
            'class': Time2VecPeriodicEncoder,
            'kwargs': {'periodic_features': periodic_features, 'd_embed': 16},
            'desc': 'Time2Vec: 1 linear + 15 sin',
        }),
    ])


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_single_dataset(dataset_key, dataset_cfg, data_dir):
    """Run full experiment for one dataset."""

    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_cfg['name']} ({dataset_key})")
    print(f"{'='*70}")

    # Load data
    data, n_rows = DATASET_LOADERS[dataset_key](data_dir)

    # Build configs
    encoding_configs = build_encoding_configs(dataset_cfg['periodic_features'])
    split_configs = dataset_cfg['splits']
    split_names = list(split_configs.keys())

    # Print split sizes
    sample_splits = create_splits(data, n_rows, split_configs, seed=42)
    print(f"\nSplit sizes:")
    for split_name, split_idx in sample_splits.items():
        print(f"  {split_name:15s}: train={len(split_idx['train']):6d}, test={len(split_idx['test']):5d}")

    # Print encoder info
    print(f"\nEncoding configurations:")
    for name, cfg in encoding_configs.items():
        enc = cfg['class'](**cfg['kwargs'])
        print(f"  {name:10s}: periodic_dim={enc.output_dim:3d}")

    # Results storage
    results = {config: {split: [] for split in split_names}
               for config in encoding_configs}

    t_start = time.time()
    total_runs = len(encoding_configs) * len(split_names) * len(SEEDS)
    run_count = 0

    for config_name, config_def in encoding_configs.items():
        print(f"\n  Config: {config_name}")

        for split_name in split_names:
            print(f"    Split: {split_name}")

            for seed in SEEDS:
                run_count += 1
                torch.manual_seed(seed)
                np.random.seed(seed)

                # Create splits
                splits = create_splits(data, n_rows, split_configs, seed)
                train_idx = splits[split_name]['train']
                test_idx = splits[split_name]['test']

                # Normalize numeric features (fit on train only)
                data_norm, _ = normalize_numeric_features(
                    data, train_idx, dataset_cfg['numeric_features']
                )

                # Create encoder and model
                encoder = config_def['class'](**config_def['kwargs'])
                nonperiodic_enc = NonPeriodicEncoder(
                    numeric_features=dataset_cfg['numeric_features'],
                    categorical_features=dataset_cfg['categorical_features'],
                    binary_features=dataset_cfg['binary_features'],
                )
                model = TabularMLPClassifier(
                    encoder, nonperiodic_enc,
                    n_classes=dataset_cfg['n_classes']
                )
                n_params = sum(p.numel() for p in model.parameters())

                # Train
                print(f"      Seed {seed} ({run_count}/{total_runs}, params={n_params}):")
                train_model(model, data_norm, train_idx, dataset_cfg,
                           epochs=100, batch_size=256)

                # Evaluate
                metrics = evaluate_model(model, data_norm, test_idx, dataset_cfg)

                result = {
                    'accuracy': metrics['accuracy'],
                    'macro_f1': metrics['macro_f1'],
                    'n_params': n_params,
                    'seed': seed,
                }
                results[config_name][split_name].append(result)

                print(f"      -> acc={metrics['accuracy']:.4f}, f1={metrics['macro_f1']:.4f}")

    total_time = time.time() - t_start

    # Aggregate
    summary = {}
    for config_name in encoding_configs:
        summary[config_name] = {}
        for split_name in split_names:
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

    # Print per-split tables
    print(f"\n{'='*70}")
    print(f"RESULTS: {dataset_cfg['name']}")
    print(f"{'='*70}")

    for split_name in split_names:
        print(f"\n--- {split_name.upper()} ---")
        print(f"{'Config':12s} | {'Accuracy':18s} | {'Macro F1':18s} | {'Params':>8s}")
        print("-" * 65)
        for config_name in encoding_configs:
            s = summary[config_name][split_name]
            acc_str = f"{s['accuracy_mean']:.4f} +/- {s['accuracy_std']:.4f}"
            f1_str = f"{s['f1_mean']:.4f} +/- {s['f1_std']:.4f}"
            print(f"{config_name:12s} | {acc_str:18s} | {f1_str:18s} | {s['n_params']:>8d}")

    # OOD average
    ood_splits = [s for s in split_names if s != 'random']
    if ood_splits:
        print(f"\n--- AVERAGE OOD ---")
        print(f"{'Config':12s} | {'Avg OOD Acc':12s} | {'Avg OOD F1':12s}")
        print("-" * 42)
        for config_name in encoding_configs:
            ood_accs = [summary[config_name][s]['accuracy_mean'] for s in ood_splits]
            ood_f1s = [summary[config_name][s]['f1_mean'] for s in ood_splits]
            print(f"{config_name:12s} | {np.mean(ood_accs):12.4f} | {np.mean(ood_f1s):12.4f}")

    return {
        'dataset': dataset_key,
        'name': dataset_cfg['name'],
        'n_rows': n_rows,
        'total_time': round(total_time, 1),
        'summary': summary,
        'raw_results': results,
        'split_names': split_names,
    }


def run_experiment():
    """Run full Phase 14B: 3 datasets x 6 configs x ~3 splits x 5 seeds."""

    print("=" * 70)
    print("SPIRNOR AI Phase 14B: Cross-Domain Tabular Validation")
    print("Two-Tier Framework: Does sin/cos consistently win on real-world tabular?")
    print("=" * 70)

    data_dir = os.path.dirname(os.path.abspath(__file__))
    all_results = {}

    for dataset_key, dataset_cfg in DATASET_CONFIGS.items():
        result = run_single_dataset(dataset_key, dataset_cfg, data_dir)
        all_results[dataset_key] = result

    # ============================================================
    # CROSS-DATASET ANALYSIS
    # ============================================================

    print("\n" + "=" * 70)
    print("CROSS-DATASET ANALYSIS")
    print("=" * 70)

    config_names = ['spirnor', 'onehot', 'sincos', 'learned', 'linear', 'time2vec']

    # Compute OOD accuracy ranking per dataset
    rankings = {}
    ood_accs_per_dataset = {}

    for dataset_key in DATASET_CONFIGS:
        result = all_results[dataset_key]
        summary = result['summary']
        ood_splits = [s for s in result['split_names'] if s != 'random']

        # Average OOD accuracy per config
        ood_means = {}
        for cfg in config_names:
            accs = [summary[cfg][s]['accuracy_mean'] for s in ood_splits]
            ood_means[cfg] = np.mean(accs)

        ood_accs_per_dataset[dataset_key] = ood_means

        # Rank (1 = best)
        sorted_cfgs = sorted(config_names, key=lambda c: -ood_means[c])
        rank = {c: i + 1 for i, c in enumerate(sorted_cfgs)}
        rankings[dataset_key] = rank

    # Print ranking table
    print(f"\nOOD Accuracy Rankings (1=best, 6=worst):")
    print(f"{'Config':12s}", end="")
    for dk in DATASET_CONFIGS:
        print(f" | {dk:16s}", end="")
    print(f" | {'Avg Rank':10s}")
    print("-" * (12 + (16 + 3) * len(DATASET_CONFIGS) + 13))

    avg_ranks = {}
    for cfg in config_names:
        print(f"{cfg:12s}", end="")
        ranks_list = []
        for dk in DATASET_CONFIGS:
            r = rankings[dk][cfg]
            acc = ood_accs_per_dataset[dk][cfg]
            print(f" | #{r} ({acc:.3f})", end="")
            ranks_list.append(r)
        avg_rank = np.mean(ranks_list)
        avg_ranks[cfg] = avg_rank
        print(f" | {avg_rank:10.2f}")

    # Final ranking
    print(f"\nFinal OOD Ranking (by avg rank):")
    for i, (cfg, rank) in enumerate(sorted(avg_ranks.items(), key=lambda x: x[1])):
        print(f"  #{i+1}: {cfg:12s} (avg rank {rank:.2f})")

    # Friedman test
    try:
        from scipy.stats import friedmanchisquare

        # Need per-dataset OOD accuracy arrays, one per config
        # Friedman requires: k treatments (configs), n blocks (datasets)
        dataset_keys = list(DATASET_CONFIGS.keys())
        arrays = []
        for cfg in config_names:
            arr = [ood_accs_per_dataset[dk][cfg] for dk in dataset_keys]
            arrays.append(arr)

        if len(dataset_keys) >= 3:
            stat, p_val = friedmanchisquare(*arrays)
            print(f"\nFriedman test: chi2={stat:.3f}, p={p_val:.4f}")
            if p_val < 0.05:
                print("  -> Significant difference between encoding methods (p < 0.05)")
            else:
                print("  -> No significant difference detected (p >= 0.05)")
        else:
            print("\n(Friedman test requires >= 3 blocks, skipped)")
    except ImportError:
        print("\n(scipy not available, skipping Friedman test)")

    # Spearman rank correlation between datasets
    try:
        from scipy.stats import spearmanr

        dataset_keys = list(DATASET_CONFIGS.keys())
        if len(dataset_keys) >= 2:
            print(f"\nSpearman rank correlation between datasets:")
            for i in range(len(dataset_keys)):
                for j in range(i + 1, len(dataset_keys)):
                    dk1, dk2 = dataset_keys[i], dataset_keys[j]
                    ranks1 = [rankings[dk1][c] for c in config_names]
                    ranks2 = [rankings[dk2][c] for c in config_names]
                    rho, p = spearmanr(ranks1, ranks2)
                    print(f"  {dk1} vs {dk2}: rho={rho:.3f}, p={p:.4f}")
    except ImportError:
        pass

    # ============================================================
    # COMBINED WITH PHASE 14 (Bike Sharing)
    # ============================================================

    # Load Phase 14 results if available
    phase14_path = os.path.join(data_dir, 'SPIRNOR_AI_PHASE14_RESULTS.json')
    if os.path.exists(phase14_path):
        print(f"\n{'='*70}")
        print("COMBINED ANALYSIS (Phase 14 + Phase 14B = 4 datasets)")
        print(f"{'='*70}")

        with open(phase14_path, 'r') as f:
            p14 = json.load(f)

        # Extract Phase 14 OOD means
        p14_summary = p14['summary']
        p14_ood_splits = [s for s in p14['splits'] if s != 'random']
        p14_ood_means = {}
        for cfg in config_names:
            accs = [p14_summary[cfg][s]['accuracy_mean'] for s in p14_ood_splits]
            p14_ood_means[cfg] = np.mean(accs)

        # Compute combined rankings
        all_ood = dict(ood_accs_per_dataset)
        all_ood['bike_sharing'] = p14_ood_means

        all_rankings = {}
        for dk in all_ood:
            sorted_cfgs = sorted(config_names, key=lambda c: -all_ood[dk][c])
            all_rankings[dk] = {c: i + 1 for i, c in enumerate(sorted_cfgs)}

        # Print combined table
        all_dks = list(all_ood.keys())
        print(f"\nCombined OOD Rankings:")
        print(f"{'Config':12s}", end="")
        for dk in all_dks:
            short = dk[:14]
            print(f" | {short:14s}", end="")
        print(f" | {'Avg Rank':10s}")
        print("-" * (12 + (14 + 3) * len(all_dks) + 13))

        combined_avg_ranks = {}
        for cfg in config_names:
            print(f"{cfg:12s}", end="")
            ranks_list = []
            for dk in all_dks:
                r = all_rankings[dk][cfg]
                acc = all_ood[dk][cfg]
                print(f" | #{r} ({acc:.2f})", end="")
                ranks_list.append(r)
            avg_rank = np.mean(ranks_list)
            combined_avg_ranks[cfg] = avg_rank
            print(f" | {avg_rank:10.2f}")

        print(f"\nFinal Combined OOD Ranking (4 datasets):")
        for i, (cfg, rank) in enumerate(sorted(combined_avg_ranks.items(), key=lambda x: x[1])):
            print(f"  #{i+1}: {cfg:12s} (avg rank {rank:.2f})")

        # Combined Friedman test
        try:
            from scipy.stats import friedmanchisquare
            arrays = [[all_ood[dk][cfg] for dk in all_dks] for cfg in config_names]
            stat, p_val = friedmanchisquare(*arrays)
            print(f"\nCombined Friedman test: chi2={stat:.3f}, p={p_val:.4f}")
            if p_val < 0.05:
                print("  -> Significant difference between encoding methods")
        except (ImportError, ValueError):
            pass

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    output = {
        'phase': 'Phase 14B: Cross-Domain Tabular Validation',
        'datasets': {},
        'cross_dataset': {
            'ood_accs_per_dataset': {dk: {c: round(v, 4) for c, v in ood.items()}
                                      for dk, ood in ood_accs_per_dataset.items()},
            'rankings': rankings,
            'avg_ranks': {c: round(v, 2) for c, v in avg_ranks.items()},
        },
    }

    for dataset_key in DATASET_CONFIGS:
        result = all_results[dataset_key]
        output['datasets'][dataset_key] = {
            'name': result['name'],
            'n_rows': result['n_rows'],
            'total_time': result['total_time'],
            'summary': result['summary'],
            'split_names': result['split_names'],
        }

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    out_path = os.path.join(data_dir, 'SPIRNOR_AI_PHASE14B_RESULTS.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")

    return output


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    run_experiment()
