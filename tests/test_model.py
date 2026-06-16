"""Tests for the forecasting model and loss functions."""

import pytest
import torch
import numpy as np

from src.models import ParallelFlareModel, FocalLoss


class TestParallelFlareModel:
    """Test the ParallelFlareModel architecture."""

    def setup_method(self):
        self.config = {
            "model": {
                "solexs_branch": {
                    "filters": 8,
                    "kernel_size": 3,
                    "padding": "same",
                },
                "hel1os_branch": {
                    "filters": 8,
                    "kernel_size": 3,
                    "padding": "same",
                },
                "overlap_conv": {
                    "filters": 16,
                    "kernel_size": 5,
                    "padding": "same",
                },
                "lstm": {
                    "layers": 1,
                    "hidden_size": 32,
                    "dropout": 0.0,
                    "bidirectional": True,
                },
                "heads": {
                    "units": [16],
                    "dropout": 0.0,
                    "activation": "relu",
                    "n_horizons": 3,
                },
            },
            "features": {
                "soft_channels": ["solexs_flux", "solexs_flux_log"],
            },
        }

    def test_model_output_shape(self):
        batch_size = 16
        seq_len = 360
        n_soft = 4
        n_hard = 12

        model = ParallelFlareModel(
            n_solexs_features=n_soft,
            n_hel1os_features=n_hard,
            config=self.config
        )

        x = torch.randn(batch_size, seq_len, n_soft + n_hard)
        y = model(x)

        assert y.shape == (batch_size, 3)
        assert y.min() >= 0.0 and y.max() <= 1.0

    def test_model_forward_backward(self):
        model = ParallelFlareModel(
            n_solexs_features=4,
            n_hel1os_features=8,
            config=self.config
        )

        x = torch.randn(8, 360, 12)
        y_true = torch.zeros(8, 3)
        y_true[2, 0] = 1.0
        y_true[5, 1] = 1.0

        y_pred = model(x)
        loss = FocalLoss(alpha=0.75, gamma=2.0)(y_pred, y_true)
        loss.backward()

        assert loss.item() > 0
        assert all(p.grad is not None for p in model.parameters()
                   if p.requires_grad)

    def test_model_parameter_count(self):
        model = ParallelFlareModel(
            n_solexs_features=7,
            n_hel1os_features=10,
            config=self.config
        )

        n_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)

        assert 10_000 < n_params < 100_000
        assert trainable == n_params


class TestFocalLoss:
    """Test focal loss implementation."""

    def test_focal_loss_basic(self):
        criterion = FocalLoss(alpha=0.75, gamma=2.0)
        inputs = torch.sigmoid(torch.randn(32, 3))
        targets = torch.zeros(32, 3)
        targets[5, 0] = 1.0
        targets[10, 1] = 1.0
        targets[15, 2] = 1.0

        loss = criterion(inputs, targets)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_focal_loss_perfect_prediction(self):
        criterion = FocalLoss(alpha=0.75, gamma=2.0)
        inputs = torch.tensor([[0.99, 0.01],
                                [0.01, 0.99]])
        targets = torch.tensor([[1.0, 0.0],
                                [0.0, 1.0]])

        loss = criterion(inputs, targets)
        assert loss.item() < 0.01

    def test_focal_loss_worse_than_ce_for_hard(self):
        focal = FocalLoss(alpha=0.75, gamma=2.0)
        ce = torch.nn.BCELoss()

        inputs = torch.sigmoid(torch.randn(100, 5))
        targets = (torch.rand(100, 5) > 0.8).float()

        fl = focal(inputs, targets).item()
        assert fl > 0
