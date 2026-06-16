"""
PyTorch Dataset and DataLoader for flare forecasting.

Creates sliding windows from processed time series for:
    - Input: last T minutes of multi-channel features
    - Target: binary flare occurrence within forecast horizons

Based on sliding window approach from Hassani et al. 2025 (ApJS, 279, 27)
which achieved TSS=0.74 with 151,071 flare events (2003-2023).
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

logger = logging.getLogger(__name__)


class FlareWindowDataset(Dataset):
    """PyTorch dataset for flare forecasting with sliding windows.

    Each sample: (input_window, label) where:
        input_window: [T x F] multi-channel time series
        label: [H] binary vector for each forecast horizon
    """

    def __init__(self, features: np.ndarray, timestamps: pd.DatetimeIndex,
                 flare_labels: dict, config: dict,
                 transform: Optional[callable] = None):
        """
        Args:
            features: [N x F] array of features
            timestamps: pd.DatetimeIndex of length N
            flare_labels: dict mapping horizon_minutes -> binary array of length N
            config: config dict with window/stride params
            transform: optional feature transform
        """
        self.features = features
        self.timestamps = timestamps
        self.flare_labels = flare_labels
        self.config = config["data"]
        self.transform = transform

        self.window_len = int(self.config["input_window_minutes"] * 60 /
                              self._infer_cadence())
        self.stride = int(self.config["sliding_stride_minutes"] * 60 /
                          self._infer_cadence())
        self.horizons = self.config["forecast_horizons_minutes"]

        self.indices = self._compute_valid_indices()
        logger.info(
            f"Dataset: {len(self.indices)} windows, "
            f"window={self.window_len} steps, "
            f"features={features.shape[1]}, "
            f"horizons={self.horizons}"
        )

    def _infer_cadence(self) -> float:
        if len(self.timestamps) < 2:
            return 60.0
        diffs = pd.Series(self.timestamps).diff().dt.total_seconds().dropna()
        return max(diffs.median(), 1.0)

    def _compute_valid_indices(self) -> List[int]:
        indices = []
        n = len(self.features)
        for i in range(0, n - self.window_len, self.stride):
            end = i + self.window_len
            if end + max(self.horizons) <= n:
                indices.append(i)
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        end = start + self.window_len

        x = self.features[start:end].astype(np.float32)

        labels = []
        for h in self.horizons:
            h_steps = int(h * 60 / self._infer_cadence())
            label_end = min(end + h_steps, len(self.features))
            label_window = self.flare_labels[h][end:label_end]
            labels.append(1.0 if label_window.max() > 0.5 else 0.0)

        y = np.array(labels, dtype=np.float32)

        if self.transform:
            x = self.transform(x)

        return torch.from_numpy(x), torch.from_numpy(y)


def create_labels_from_flares(features_df: pd.DataFrame,
                              flare_catalogue: pd.DataFrame,
                              horizons_minutes: List[int]) -> dict:
    """Create binary labels for multiple forecast horizons.

    For each time step, label=1 if a flare peaks within any horizon window.
    """
    labels = {h: np.zeros(len(features_df), dtype=np.float32)
              for h in horizons_minutes}

    times = features_df.index

    for _, flare in flare_catalogue.iterrows():
        peak_time = flare["peak_time"]
        for h in horizons_minutes:
            start_idx = np.searchsorted(times, peak_time - pd.Timedelta(minutes=h))
            end_idx = np.searchsorted(times, peak_time)
            labels[h][start_idx:end_idx] = 1.0

    for h in horizons_minutes:
        n_pos = labels[h].sum()
        total = len(labels[h])
        logger.info(f"Horizon {h}min: {int(n_pos)} positive / {total} total "
                    f"({n_pos/total*100:.2f}%)")

    return labels


def create_data_loaders(features: np.ndarray, timestamps: pd.DatetimeIndex,
                        labels: dict, horizons: List[int],
                        config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders with chronological split and weighting."""
    n = len(features)
    train_end = int(n * config["data"]["train_split"])
    val_end = int(n * (config["data"]["train_split"] + config["data"]["val_split"]))

    train_labels_h = {h: labels[h][:train_end] for h in horizons}
    val_labels_h = {h: labels[h][train_end:val_end] for h in horizons}
    test_labels_h = {h: labels[h][val_end:n] for h in horizons}

    train_ds = FlareWindowDataset(
        features[:train_end], timestamps[:train_end],
        train_labels_h, config
    )
    val_ds = FlareWindowDataset(
        features[train_end:val_end], timestamps[train_end:val_end],
        val_labels_h, config
    )
    test_ds = FlareWindowDataset(
        features[val_end:n], timestamps[val_end:n],
        test_labels_h, config
    )

    batch_size = config["training"]["batch_size"]

    pos_weight = _compute_pos_weight(train_ds)
    sampler = None
    if config["training"]["oversample_flares"]:
        sampler = _create_weighted_sampler(train_ds, config["training"]["oversample_factor"])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader, pos_weight


def _compute_pos_weight(dataset: FlareWindowDataset) -> float:
    """Compute ratio of negative to positive samples for loss weighting."""
    total = len(dataset)
    pos = 0
    for i in range(min(total, 1000)):
        _, y = dataset[i]
        pos += y.max().item()
    pos_frac = pos / min(total, 1000)
    if pos_frac < 0.01:
        pos_frac = 0.01
    return (1.0 - pos_frac) / pos_frac


def _create_weighted_sampler(dataset: FlareWindowDataset,
                             oversample_factor: float = 5.0) -> WeightedRandomSampler:
    """Create weighted sampler to oversample flare-positive windows."""
    weights = np.ones(len(dataset))
    for i in range(len(dataset)):
        _, y = dataset[i]
        if y.max() > 0.5:
            weights[i] = oversample_factor
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=int(len(dataset) * 0.5),
        replacement=True
    )
    return sampler
