"""Tests for data ingestion and preprocessing modules."""

import pytest
import numpy as np
import pandas as pd

from src.data.preprocessing import FlarePreprocessor, merge_soft_hard
from src.data.dataset import FlareWindowDataset, create_labels_from_flares


class TestFlarePreprocessor:
    """Test the preprocessing pipeline."""

    def setup_method(self):
        self.config = {
            "preprocessing": {
                "binning_cadence": 10,
                "log_transform": True,
                "standardize": True,
                "fill_method": "ffill",
                "max_gap_seconds": 300,
                "background_percentile": 10,
                "smoothing_window_minutes": 1,
            },
            "features": {
                "soft_channels": ["solexs_flux", "solexs_flux_log",
                                   "dsoft_dt_30s", "background_subtracted_soft"],
                "hard_channels": ["hel1os_flux", "hel1os_flux_log",
                                   "spectral_hardness_ratio", "dhard_dt_30s",
                                   "overlap_xcorr", "background_subtracted_hard"],
            }
        }

    def test_compute_features_has_all_columns(self):
        n = 1000
        df = pd.DataFrame({
            "solexs_flux": 1e-6 + np.random.randn(n) * 1e-7,
            "hel1os_flux": 3e-7 + np.random.randn(n) * 5e-8
        })
        preproc = FlarePreprocessor(self.config)
        result = preproc.compute_features(df)

        expected = [
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
        for feat in expected:
            assert feat in result.columns, f"Missing feature: {feat}"
        assert len(result) == n

    def test_standardize(self):
        df = pd.DataFrame({
            "solexs_flux": np.random.randn(100) * 10 + 50,
            "hel1os_flux": np.random.randn(100) * 5 + 20
        })
        preproc = FlarePreprocessor(self.config)
        result, params = preproc.standardize(df)
        for col in ["solexs_flux", "hel1os_flux"]:
            assert abs(result[col].mean()) < 1e-10
            assert abs(result[col].std() - 1.0) < 1e-6

    def test_handle_gaps(self):
        times = pd.date_range("2024-01-01", periods=100, freq="1min")
        df = pd.DataFrame({
            "solexs_flux": np.ones(100),
            "hel1os_flux": np.ones(100)
        }, index=times)
        df.iloc[50:55, df.columns.get_loc("solexs_flux")] = np.nan
        preproc = FlarePreprocessor(self.config)
        result = preproc.handle_gaps(df)
        assert result.isnull().sum().sum() == 0

    def test_merge_soft_hard(self):
        soft = pd.DataFrame({"soft_flux": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2, freq="1s"))
        hard = pd.DataFrame({"hard_flux": [0.5, 0.6]}, index=pd.date_range("2024-01-01", periods=2, freq="1s"))
        merged = merge_soft_hard(soft, hard)
        assert "solexs_flux" in merged.columns
        assert "hel1os_flux" in merged.columns
        assert merged["solexs_flux"].iloc[0] == 1.0
        assert merged["hel1os_flux"].iloc[0] == 0.5


class TestFlareWindowDataset:
    """Test the sliding window dataset."""

    def test_dataset_creation(self):
        n_frames = 500
        features = np.random.randn(n_frames, 16).astype(np.float32)
        timestamps = pd.date_range("2024-01-01", periods=n_frames, freq="10s")
        labels = {h: np.zeros(n_frames) for h in [15, 30, 60]}
        labels[15][200:250] = 1.0

        config = {
            "data": {
                "input_window_minutes": 60,
                "forecast_horizons_minutes": [15, 30, 60],
                "sliding_stride_minutes": 5,
            }
        }

        ds = FlareWindowDataset(features, timestamps, labels, config)
        assert len(ds) > 0

        x, y = ds[0]
        assert x.shape[0] == 360
        assert x.shape[1] == 16
        assert y.shape[0] == 3

    def test_labels_creation(self):
        n = 1000
        times = pd.date_range("2024-01-01", periods=n, freq="10s")
        df = pd.DataFrame({"flux": np.ones(n)}, index=times)
        flares = pd.DataFrame({
            "start_time": [times[300]],
            "peak_time": [times[350]],
            "end_time": [times[400]],
            "goes_class": ["M1.0"],
            "peak_flux": [1e-4]
        })
        labels = create_labels_from_flares(df, flares, [15, 30, 60])
        for h in [15, 30, 60]:
            assert h in labels
            assert labels[h].sum() > 0
            assert labels[h].dtype == np.float32
