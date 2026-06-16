"""
Parallel 1D-CNN-BiLSTM-MultiHead model for solar flare forecasting.

Architecture (optimized for small Aditya-L1 dataset, July 2024 onward):

    [Input: B x T x F]
         |
    +----+----+
    |         |
  SoLEXS   HEL1OS        <- Parallel branches (sensor-specific noise)
    |         |
  1D-Conv   1D-Conv
  (32,k=3)  (32,k=3)
    |         |
    +----+----+
         |
   Overlap Conv          <- 64 filters, k=5 (8-22 keV cross-correlation)
         |
   BiLSTM (64)           <- Single layer, bidirectional
         |
    +----+----+
    |    |    |
  Head1 Head2 Head3      <- Multi-head: 15, 30, 60 min
  Dense(32) -> Drop(0.4) -> Dense(1, sigmoid)

Key design decisions:
    - Parallel branches preserve per-instrument noise distributions
    - Overlap Conv (k=5) learns non-linear cross-correlations in 8-22 keV band
    - Single BiLSTM(64) prevents overfitting on small data
    - Multi-head outputs give independent probabilities per horizon
    - High dropout (0.4) and L2 regularization for small dataset

Target: TSS >= 0.65 on >=C-class flares
Based on: Hassani et al. 2025 (TSS=0.74 benchmark, GOES 2003-2023)
"""

import logging
from typing import List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ParallelFlareModel(nn.Module):
    """Parallel 1D-CNN-BiLSTM with multi-head output for flare forecasting.

    Input:  [B, T, F] where F = SoLEXS features + HEL1OS features + derived
    Output: [B, 3] sigmoid probabilities for [15min, 30min, 60min]
    """

    def __init__(self, n_solexs_features: int, n_hel1os_features: int,
                 config: dict):
        super().__init__()
        model_cfg = config["model"]

        # --- Parallel instrument branches ---
        solexs_cfg = model_cfg["solexs_branch"]
        self.solexs_conv = nn.Sequential(
            nn.Conv1d(n_solexs_features, solexs_cfg["filters"],
                      kernel_size=solexs_cfg["kernel_size"],
                      padding=solexs_cfg["padding"]),
            nn.BatchNorm1d(solexs_cfg["filters"]),
            nn.ReLU(inplace=True),
        )

        hel1os_cfg = model_cfg["hel1os_branch"]
        self.hel1os_conv = nn.Sequential(
            nn.Conv1d(n_hel1os_features, hel1os_cfg["filters"],
                      kernel_size=hel1os_cfg["kernel_size"],
                      padding=hel1os_cfg["padding"]),
            nn.BatchNorm1d(hel1os_cfg["filters"]),
            nn.ReLU(inplace=True),
        )

        # --- Overlap synthesis convolution ---
        overlap_cfg = model_cfg["overlap_conv"]
        branch_out = solexs_cfg["filters"] + hel1os_cfg["filters"]
        self.overlap_conv = nn.Sequential(
            nn.Conv1d(branch_out, overlap_cfg["filters"],
                      kernel_size=overlap_cfg["kernel_size"],
                      padding=overlap_cfg["padding"]),
            nn.BatchNorm1d(overlap_cfg["filters"]),
            nn.ReLU(inplace=True),
        )

        # --- BiLSTM ---
        lstm_cfg = model_cfg["lstm"]
        self.lstm = nn.LSTM(
            input_size=overlap_cfg["filters"],
            hidden_size=lstm_cfg["hidden_size"],
            num_layers=lstm_cfg["layers"],
            batch_first=True,
            dropout=0.0,
            bidirectional=lstm_cfg["bidirectional"]
        )
        lstm_out = lstm_cfg["hidden_size"] * (2 if lstm_cfg["bidirectional"] else 1)

        # --- Multi-head output ---
        heads_cfg = model_cfg["heads"]
        self.heads = nn.ModuleList()
        for _ in range(heads_cfg["n_horizons"]):
            head = nn.Sequential(
                nn.Linear(lstm_out, heads_cfg["units"][0]),
                nn.ReLU(inplace=True),
                nn.Dropout(heads_cfg["dropout"]),
                nn.Linear(heads_cfg["units"][0], 1),
            )
            self.heads.append(head)

        self._init_weights()
        self._log_architecture(n_solexs_features, n_hel1os_features, config)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def _log_architecture(self, n_soft, n_hard, config):
        n_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("=" * 50)
        logger.info("ParallelFlareModel Architecture")
        logger.info("=" * 50)
        logger.info(f"  SoLEXS branch: Conv1D({n_soft}->32, k=3)")
        logger.info(f"  HEL1OS branch: Conv1D({n_hard}->32, k=3)")
        logger.info(f"  Overlap Conv:  Conv1D(64->64, k=5)")
        logger.info(f"  BiLSTM:        hidden=64, 1-layer, bidirectional")
        logger.info(f"  Output heads:  3 independent Dense(32->1, sigmoid)")
        logger.info(f"  Dropout:       {config['model']['heads']['dropout']}")
        logger.info(f"  Total params:  {n_params:,}")
        logger.info(f"  Trainable:     {trainable:,}")
        logger.info("=" * 50)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the parallel architecture.

        Args:
            x: [B, T, F] — full feature tensor
               Expects first n_soft channels as SoLEXS, rest as HEL1OS
        Returns:
            [B, 3] — sigmoid probabilities for 15, 30, 60 min horizons
        """
        B, T, F = x.shape

        n_soft = self.solexs_conv[0].in_channels
        x_soft = x[:, :, :n_soft]
        x_hard = x[:, :, n_soft:]

        x_soft = x_soft.transpose(1, 2)
        x_hard = x_hard.transpose(1, 2)

        soft_out = self.solexs_conv(x_soft)
        hard_out = self.hel1os_conv(x_hard)

        combined = torch.cat([soft_out, hard_out], dim=1)

        overlap_out = self.overlap_conv(combined)
        overlap_out = overlap_out.transpose(1, 2)

        lstm_out, _ = self.lstm(overlap_out)
        last_step = lstm_out[:, -1, :]

        outputs = []
        for head in self.heads:
            outputs.append(head(last_step))

        return torch.sigmoid(torch.cat(outputs, dim=1))

    def get_feature_split(self, total_features: int) -> tuple:
        """Return how features are split between SoLEXS and HEL1OS branches."""
        n_soft = self.solexs_conv[0].in_channels
        n_hard = total_features - n_soft
        return n_soft, n_hard


class FlareForecastModel(ParallelFlareModel):
    """Alias for backward compatibility — delegates to ParallelFlareModel."""
    pass


def split_features_for_branches(n_features: int,
                                 config: dict) -> tuple:
    """Determine feature split between SoLEXS and HEL1OS branches.

    Returns: (n_solexs_features, n_hel1os_features)
    """
    feat_cfg = config["features"]
    n_soft = len(feat_cfg.get("soft_channels", ["solexs_flux"]))
    n_hard = n_features - n_soft
    return n_soft, max(n_hard, 1)
