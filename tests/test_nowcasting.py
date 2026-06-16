"""Tests for the nowcasting detection module."""

import pytest
import numpy as np
import pandas as pd

from src.nowcasting import ThresholdFlareDetector, Nowcaster, FlareCatalogueMerger


class TestThresholdFlareDetector:
    """Test threshold-based flare detection."""

    def setup_method(self):
        self.config = {
            "nowcasting": {
                "soft_channel_threshold_sigma": 3.0,
                "hard_channel_threshold_sigma": 3.0,
                "dFdt_threshold_sigma": 2.0,
                "consecutive_samples": 3,
                "cooldown_minutes": 15,
                "merge_window_seconds": 60,
            }
        }

    def test_detect_flare(self):
        np.random.seed(42)
        n = 1000
        times = pd.date_range("2024-01-01", periods=n, freq="1s")
        flux = np.ones(n) * 1e-7 + np.random.randn(n) * 1e-8
        flux[500:600] = 1e-5 + np.random.randn(100) * 1e-6

        detector = ThresholdFlareDetector("test", self.config)
        results = detector.detect(times, flux)

        assert len(results) >= 1
        assert "peak_time" in results.columns
        assert "goes_class" in results.columns

    def test_classify_flare(self):
        detector = ThresholdFlareDetector("test", self.config)
        assert detector._classify_flare(1e-8) == "A"
        assert detector._classify_flare(5e-7) == "B"
        assert detector._classify_flare(5e-6) == "C"
        assert detector._classify_flare(5e-5) == "M"
        assert detector._classify_flare(5e-4) == "X"

    def test_no_detection_on_quiet_data(self):
        np.random.seed(42)
        n = 500
        times = pd.date_range("2024-01-01", periods=n, freq="1s")
        flux = np.ones(n) * 1e-8 + np.random.randn(n) * 1e-10

        detector = ThresholdFlareDetector("quiet_test", self.config)
        results = detector.detect(times, flux)

        assert len(results) == 0


class TestFlareCatalogueMerger:
    """Test catalogue merging."""

    def test_merge_two_catalogues(self):
        cat1 = pd.DataFrame({
            "start_time": pd.to_datetime(["2024-01-01 00:00:00"]),
            "peak_time": pd.to_datetime(["2024-01-01 00:05:00"]),
            "end_time": pd.to_datetime(["2024-01-01 00:10:00"]),
            "peak_flux": [1e-5],
            "goes_class": ["C"],
            "source": ["SoLEXS"]
        })
        cat2 = pd.DataFrame({
            "start_time": pd.to_datetime(["2024-01-01 00:01:00"]),
            "peak_time": pd.to_datetime(["2024-01-01 00:05:30"]),
            "end_time": pd.to_datetime(["2024-01-01 00:11:00"]),
            "peak_flux": [1.5e-5],
            "goes_class": ["C"],
            "source": ["HEL1OS"]
        })

        merger = FlareCatalogueMerger(merge_window_seconds=120)
        merged = merger.merge([cat1, cat2], ["SoLEXS", "HEL1OS"])
        assert len(merged) == 1
        assert "SoLEXS" in merged.iloc[0]["sources"]
        assert "HEL1OS" in merged.iloc[0]["sources"]
