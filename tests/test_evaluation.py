"""Tests for the evaluation metrics module."""

import pytest
import numpy as np

from src.evaluation import FlareEvaluationMetrics


class TestFlareEvaluationMetrics:
    """Test evaluation metrics computation."""

    def setup_method(self):
        self.metrics = FlareEvaluationMetrics(thresholds=[0.3, 0.5, 0.7])

    def test_perfect_prediction(self):
        y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.float32)
        y_pred = np.array([[0.99, 0.01], [0.01, 0.99],
                           [0.99, 0.99], [0.01, 0.01]], dtype=np.float32)

        results = self.metrics.compute_all(y_true, y_pred, ["15min", "30min"])
        for h in ["15min", "30min"]:
            assert results[h]["best_tss"] > 0.9
            assert results[h]["auc"] > 0.9

    def test_random_prediction(self):
        np.random.seed(42)
        y_true = np.zeros((1000, 3), dtype=np.float32)
        y_true[100:200, 0] = 1.0
        y_true[300:350, 1] = 1.0
        y_true[500:550, 2] = 1.0

        y_pred = np.random.rand(1000, 3)

        results = self.metrics.compute_all(y_true, y_pred, ["15min", "30min", "60min"])
        for h in ["15min", "30min", "60min"]:
            assert results[h]["best_tss"] < 0.5
            assert 0.35 <= results[h]["auc"] <= 0.65

    def test_tss_imbalance_invariant(self):
        """TSS should not be affected by class imbalance."""
        n = 10000
        y_true = np.zeros(n, dtype=np.float32)
        y_true[:50] = 1.0
        y_pred = np.zeros(n, dtype=np.float32)
        y_pred[:50] = 0.99
        y_pred[50:100] = 0.01

        tss = self.metrics.compute_tss(y_true, y_pred, threshold=0.5)
        assert abs(tss - 1.0) < 0.01

        all_neg = np.zeros(n, dtype=np.float32)
        all_pred_neg = np.zeros(n, dtype=np.float32)
        tss_all_neg = self.metrics.compute_tss(all_neg, all_pred_neg, threshold=0.5)
        assert tss_all_neg == 0.0

    def test_lead_time(self):
        forecast_times = np.array([
            np.datetime64("2024-01-01T00:05:00"),
            np.datetime64("2024-01-01T01:00:00"),
        ])
        flare_times = np.array([
            np.datetime64("2024-01-01T00:15:00"),
            np.datetime64("2024-01-01T01:30:00"),
        ])

        lt = self.metrics.compute_lead_time(forecast_times, flare_times)
        assert lt["mean_lead_time"] > 0
        assert lt["n_triggers"] == 2
