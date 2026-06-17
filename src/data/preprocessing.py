"""
Preprocessing pipeline for Aditya-L1 SoLEXS + HEL1OS X-ray light curves.

Key design for 8-22 keV overlap optimization:
    1. 1s -> 10s binning (downsample by block-mean)
    2. Spectral Hardness Ratio: HEL1OS(8-22) / SoLEXS(2-8)
    3. Flux derivatives at 30s and 2min windows
    4. Rolling Pearson cross-correlation in overlap band
    5. Background subtraction via trailing percentile
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal, ndimage, stats

logger = logging.getLogger(__name__)


class FlarePreprocessor:
    """Preprocesses SoLEXS + HEL1OS light curves with 8-22 keV overlap focus."""

    def __init__(self, config: dict):
        self.cfg = config.get("preprocessing", config)
        self.feat_cfg = config.get("features", config)
        self.binning = self.cfg.get("binning_cadence", 10)
        self.bg_percentile = self.cfg.get("background_percentile", 10)

    def bin_to_cadence(self, df: pd.DataFrame,
                       target_cadence_s: int = None) -> pd.DataFrame:
        """Downsample from 1s native cadence to target cadence (default 10s).

        Uses block-mean to preserve flux statistics.
        60-min window at 10s = 360 timesteps (tractable for LSTM).
        """
        if target_cadence_s is None:
            target_cadence_s = self.binning
        return df.resample(f"{target_cadence_s}s").mean()

    def unify_and_bin(self, solexs: pd.DataFrame, hel1os: pd.DataFrame,
                      target_cadence_s: int = 10) -> pd.DataFrame:
        """Unify SoLEXS and HEL1OS to common 10s time grid.

        Handles non-overlapping data by forward-filling missing instruments
        with zeros and marking the is_valid column accordingly.
        """
        soft = self.bin_to_cadence(solexs, target_cadence_s)
        hard = self.bin_to_cadence(hel1os, target_cadence_s)

        # Check temporal overlap
        if not soft.empty and not hard.empty:
            s_start, s_end = soft.index.min(), soft.index.max()
            h_start, h_end = hard.index.min(), hard.index.max()
            overlap_start = max(s_start, h_start)
            overlap_end = min(s_end, h_end)
            has_overlap = overlap_start < overlap_end
            if not has_overlap:
                logger.warning(
                    f"No temporal overlap between SoLEXS ({s_start} to {s_end}) "
                    f"and HEL1OS ({h_start} to {h_end}). "
                    f"Proceeding with single-instrument fallback."
                )

        combined = soft.join(hard, how="outer")
        # Fill missing instrument data with 0 (not ffill — that would create artifacts)
        combined["solexs_flux"] = combined["solexs_flux"].fillna(0)
        combined["hel1os_flux"] = combined["hel1os_flux"].fillna(0)

        if "solexs_flux" not in combined.columns and len(combined.columns) >= 1:
            combined.columns = ["solexs_flux"] + list(combined.columns[1:])
        if "hel1os_flux" not in combined.columns and len(combined.columns) >= 2:
            combined.columns = list(combined.columns[:1]) + ["hel1os_flux"] + list(combined.columns[2:])

        return combined

    def handle_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and fill data gaps."""
        max_gap = self.cfg["max_gap_seconds"]
        if isinstance(df.index, pd.DatetimeIndex):
            diffs = df.index.to_series().diff().dt.total_seconds()
            large_gaps = diffs > max_gap
            if large_gaps.any():
                logger.warning(f"Detected {large_gaps.sum()} large gaps")
            fill_method = self.cfg.get("fill_method", "ffill")
            df = df.ffill() if fill_method == "ffill" else df.interpolate(method="time")
            df["is_valid"] = (~large_gaps).astype(float)
        else:
            df["is_valid"] = 1.0
        return df

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute overlap-optimized physics features.

        Feature set targets the 8-22 keV overlap band where pre-flare
        precursor brightening may appear.

        NOTE: spectral_hardness_ratio = HEL1OS(8-150 keV) / SoLEXS(2-22 keV).
        This is an instrument-dependent ratio, not a true physical hardness ratio.
        It captures instrument response differences as well as spectral changes.
        A true hardness ratio would require overlapping energy bands from one instrument.

        Features:
            - Soft/hard flux (log-transformed)
            - Spectral Hardness Ratio (instrument-dependent, see note above)
            - dF/dt at 30s and 2min windows
            - Rolling Pearson correlation between instruments
            - Rolling statistics (1-min mean/std)
            - Background-subtracted fluxes
        """
        result = df.copy()
        soft = result["solexs_flux"].values.astype(np.float64)
        hard = result["hel1os_flux"].values.astype(np.float64)

        eps = 1e-12
        soft_safe = np.maximum(soft, eps)
        hard_safe = np.maximum(hard, eps)

        result["solexs_flux_log"] = np.log(soft_safe)
        result["hel1os_flux_log"] = np.log(hard_safe)

        cadence_s = self._infer_cadence(result)

        # Spectral Hardness Ratio: HEL1OS / SoLEXS
        result["spectral_hardness_ratio"] = hard_safe / soft_safe

        # Flux derivatives at multiple windows
        window_30s = max(1, int(30 / cadence_s))
        window_2min = max(1, int(120 / cadence_s))
        window_1min = max(1, int(60 / cadence_s))

        def rolling_gradient(arr, w):
            padded = np.pad(arr, (w, w), mode="edge")
            grad = np.gradient(padded, cadence_s)
            return grad[w:-w]

        dsoft_30s = rolling_gradient(soft, window_30s)
        dsoft_2min = rolling_gradient(soft, window_2min)
        dhard_30s = rolling_gradient(hard, window_30s)
        dhard_2min = rolling_gradient(hard, window_2min)

        result["dsoft_dt_30s"] = dsoft_30s
        result["dsoft_dt_2min"] = dsoft_2min
        result["dhard_dt_30s"] = dhard_30s
        result["dhard_dt_2min"] = dhard_2min

        # Rolling Pearson cross-correlation in overlap band
        result["overlap_xcorr"] = self._rolling_pearson(
            soft, hard, window=window_2min
        )

        # Rolling statistics (1-min windows)
        result["soft_rolling_mean_1min"] = self._rolling_mean(soft, window_1min)
        result["soft_rolling_std_1min"] = self._rolling_std(soft, window_1min)
        result["hard_rolling_mean_1min"] = self._rolling_mean(hard, window_1min)
        result["hard_rolling_std_1min"] = self._rolling_std(hard, window_1min)

        result["hard_rolling_mean_5min"] = self._rolling_mean(
            hard, max(1, int(300 / cadence_s))
        )

        # Background-subtracted fluxes
        bg_soft = np.percentile(soft, self.bg_percentile)
        bg_hard = np.percentile(hard, self.bg_percentile)
        result["background_subtracted_soft"] = soft - bg_soft
        result["background_subtracted_hard"] = hard - bg_hard

        # If one instrument is all zeros (no data), zero out its derived features
        # to prevent garbage features from forward-filled/zero-filled data
        if np.all(soft == 0):
            for col in ["solexs_flux_log", "dsoft_dt_30s", "dsoft_dt_2min",
                        "soft_rolling_mean_1min", "soft_rolling_std_1min",
                        "background_subtracted_soft"]:
                result[col] = 0.0
            result["spectral_hardness_ratio"] = 0.0
            result["overlap_xcorr"] = 0.0
        if np.all(hard == 0):
            for col in ["hel1os_flux_log", "dhard_dt_30s", "dhard_dt_2min",
                        "hard_rolling_mean_1min", "hard_rolling_std_1min",
                        "hard_rolling_mean_5min", "background_subtracted_hard"]:
                result[col] = 0.0
            result["spectral_hardness_ratio"] = 0.0
            result["overlap_xcorr"] = 0.0

        valid_cols = [
            "solexs_flux", "hel1os_flux",
            "solexs_flux_log", "hel1os_flux_log",
            "spectral_hardness_ratio",
            "dsoft_dt_30s", "dsoft_dt_2min",
            "dhard_dt_30s", "dhard_dt_2min",
            "overlap_xcorr",
            "soft_rolling_mean_1min", "soft_rolling_std_1min",
            "hard_rolling_mean_1min", "hard_rolling_std_1min",
            "hard_rolling_mean_5min",
            "background_subtracted_soft", "background_subtracted_hard",
            "is_valid"
        ]
        for col in valid_cols:
            if col not in result.columns:
                result[col] = 0.0

        return result[valid_cols].copy()

    def standardize(self, df: pd.DataFrame,
                    fit_params: Optional[dict] = None
                    ) -> Tuple[pd.DataFrame, dict]:
        """Z-score standardization. Returns (df, params)."""
        exclude = {"is_valid"}
        feature_cols = [c for c in df.columns if c not in exclude]
        result = df.copy()

        if fit_params is None:
            fit_params = {}
            for col in feature_cols:
                mean = df[col].mean()
                std = df[col].std()
                if std < 1e-12:
                    std = 1.0
                fit_params[col] = {"mean": float(mean), "std": float(std)}
                result[col] = (df[col] - mean) / std
        else:
            for col in feature_cols:
                if col in fit_params:
                    p = fit_params[col]
                    result[col] = (df[col] - p["mean"]) / p["std"]
        return result, fit_params

    def _infer_cadence(self, df: pd.DataFrame) -> float:
        if not isinstance(df.index, pd.DatetimeIndex):
            return float(self.binning)
        diffs = df.index.to_series().diff().dropna()
        if len(diffs) == 0:
            return float(self.binning)
        if hasattr(diffs, "dt"):
            diffs = diffs.dt.total_seconds()
        elif hasattr(diffs.iloc[0], "total_seconds"):
            diffs = diffs.apply(lambda x: x.total_seconds())
        else:
            return float(self.binning)
        return max(diffs.median(), 1.0)

    def _rolling_mean(self, arr, window):
        if window < 1:
            window = 1
        return ndimage.uniform_filter1d(arr.astype(np.float64),
                                        size=window, mode="nearest")

    def _rolling_std(self, arr, window):
        if window < 1:
            window = 1
        result = np.zeros_like(arr)
        for i in range(len(arr)):
            start = max(0, i - window // 2)
            end = min(len(arr), i + window // 2 + 1)
            result[i] = np.std(arr[start:end])
        return result

    def _rolling_pearson(self, arr1: np.ndarray, arr2: np.ndarray,
                         window: int) -> np.ndarray:
        """Rolling Pearson correlation between two arrays."""
        result = np.zeros(len(arr1))
        half = max(1, window // 2)
        for i in range(len(arr1)):
            start = max(0, i - half)
            end = min(len(arr1), i + half + 1)
            a = arr1[start:end]
            b = arr2[start:end]
            if len(a) < 3:
                result[i] = 0.0
                continue
            r, _ = stats.pearsonr(a, b)
            result[i] = 0.0 if np.isnan(r) else np.clip(r, -1.0, 1.0)
        return result


def merge_soft_hard(soft_df: pd.DataFrame, hard_df: pd.DataFrame) -> pd.DataFrame:
    """Merge soft and hard flux DataFrames into a unified DataFrame.

    Renames columns to solexs_flux and hel1os_flux for preprocessing compat.
    """
    result = pd.DataFrame(index=soft_df.index)
    if "solexs_flux" in soft_df.columns:
        result["solexs_flux"] = soft_df["solexs_flux"]
    elif "soft_flux" in soft_df.columns:
        result["solexs_flux"] = soft_df["soft_flux"]
    else:
        result["solexs_flux"] = soft_df.iloc[:, 0]

    if "hel1os_flux" in hard_df.columns:
        result["hel1os_flux"] = hard_df["hel1os_flux"]
    elif "hard_flux" in hard_df.columns:
        result["hel1os_flux"] = hard_df["hard_flux"]
    else:
        result["hel1os_flux"] = hard_df.iloc[:, 0]

    return result
