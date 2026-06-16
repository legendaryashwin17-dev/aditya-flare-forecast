"""
Training pipeline for ParallelFlareModel.

Optimized for small Aditya-L1 dataset:
    - Focal Loss (alpha=0.75, gamma=2.0) for class imbalance
    - Adam lr=1e-3 with ReduceLROnPlateau (factor=0.5, patience=3)
    - Early stopping monitoring val TSS (patience=10)
    - Hyperbolic tangent or clipped gradients for stability
    - Transfer learning: freeze branches, fine-tune overlap+LSTM+heads

Target: TSS >= 0.65 on >=C-class flares
"""

import os
import logging
from typing import Dict, Optional, Tuple
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models import ParallelFlareModel, FocalLoss
from src.evaluation import FlareEvaluationMetrics

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for the parallel flare forecasting model."""

    def __init__(self, model: ParallelFlareModel, config: dict,
                 device: torch.device = None):
        self.model = model
        self.cfg = config["training"]
        self.model_cfg = config["model"]

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available()
                                       else "cpu")
        else:
            self.device = device

        self.model = self.model.to(self.device)
        logger.info(f"Training on device: {self.device}")

        self.criterion = FocalLoss(
            alpha=self.cfg.get("focal_alpha", 0.75),
            gamma=self.cfg.get("focal_gamma", 2.0)
        )

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.cfg.get("learning_rate", 0.001),
            weight_decay=self.cfg.get("weight_decay", 0.0001)
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=self.cfg.get("lr_factor", 0.5),
            patience=self.cfg.get("lr_patience", 3),
            min_lr=1e-6
        )

        self.metrics_eval = FlareEvaluationMetrics(
            thresholds=[0.3, 0.5, 0.7]
        )

        self.best_val_tss = -1.0
        self.best_model_state = None
        self.best_epoch = 0
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.val_tss_scores = []

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        all_y_true = []
        all_y_pred = []

        for x, y in val_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)

            total_loss += loss.item()
            n_batches += 1

            all_y_true.append(y.cpu().numpy())
            all_y_pred.append(y_pred.cpu().numpy())

        avg_loss = total_loss / max(n_batches, 1)

        y_true = np.concatenate(all_y_true, axis=0)
        y_pred = np.concatenate(all_y_pred, axis=0)

        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        horizons = [15, 30, 60]
        horizon_labels = [f"{h}min" for h in horizons]
        metrics = self.metrics_eval.compute_all(y_true, y_pred, horizon_labels)

        return avg_loss, metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            max_epochs: int = None,
            checkpoint_dir: Optional[str] = None) -> Dict:
        if max_epochs is None:
            max_epochs = self.cfg.get("max_epochs", 100)

        patience = self.cfg.get("early_stopping_patience", 10)
        logger.info(f"Training up to {max_epochs} epochs, patience={patience}")
        logger.info(f"Optimizer: Adam(lr={self.cfg.get('learning_rate', 0.001)}, "
                    f"wd={self.cfg.get('weight_decay', 0.0001)})")
        logger.info(f"LR scheduler: ReduceLROnPlateau(factor=0.5, patience=3)")
        logger.info(f"Early stopping: monitor val TSS, patience={patience}")

        for epoch in range(1, max_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics = self.validate(val_loader)

            val_tss = val_metrics.get("mean_tss_across_horizons", 0.0)
            val_auc = val_metrics.get("mean_auc_across_horizons", 0.0)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_tss_scores.append(val_tss)

            self.scheduler.step(val_tss)

            if epoch == 1 or epoch % 5 == 0 or val_tss > self.best_val_tss:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch:3d}/{max_epochs} | "
                    f"Train: {train_loss:.4f} | "
                    f"Val: {val_loss:.4f} | "
                    f"TSS: {val_tss:.4f} | "
                    f"AUC: {val_auc:.4f} | "
                    f"LR: {lr:.2e}"
                )

            if val_tss > self.best_val_tss:
                self.best_val_tss = val_tss
                self.best_epoch = epoch
                self.best_model_state = deepcopy(self.model.state_dict())
                self.patience_counter = 0

                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    path = os.path.join(checkpoint_dir, "best_model.pt")
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_tss": val_tss,
                        "val_auc": val_auc,
                        "config": self.cfg
                    }, path)
                    logger.info(f"  -> Saved best model (TSS={val_tss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info(
                        f"Early stopping at epoch {epoch}. "
                        f"Best TSS={self.best_val_tss:.4f} at epoch "
                        f"{self.best_epoch}"
                    )
                    break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        logger.info("=" * 50)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Best epoch: {self.best_epoch}")
        logger.info(f"Best val TSS: {self.best_val_tss:.4f}")
        logger.info("=" * 50)

        return {
            "best_val_tss": self.best_val_tss,
            "best_epoch": self.best_epoch,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_tss_scores": self.val_tss_scores
        }

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Evaluate on held-out test set."""
        logger.info("Evaluating on test set...")
        _, metrics = self.validate(test_loader)

        for h, m in metrics.items():
            if isinstance(m, dict):
                tss = m.get("best_tss", 0)
                auc = m.get("auc", 0)
                lt = m.get("mean_lead_time", None)
                lt_str = f", lead={lt:.1f}min" if lt is not None else ""
                logger.info(f"  {h}: TSS={tss:.4f}, AUC={auc:.4f}{lt_str}")

        logger.info(f"  Mean TSS: {metrics.get('mean_tss_across_horizons', 0):.4f}")
        logger.info(f"  Mean AUC: {metrics.get('mean_auc_across_horizons', 0):.4f}")
        return metrics

    def freeze_branches_for_finetune(self):
        """Freeze SoLEXS and HEL1OS branch CNNs for transfer learning.

        Keeps overlap Conv + BiLSTM + heads trainable.
        """
        frozen = 0
        total = 0
        for name, param in self.model.named_parameters():
            if "solexs_conv" in name or "hel1os_conv" in name:
                param.requires_grad = False
                frozen += 1
            total += 1

        logger.info(f"Transfer learning: frozen {frozen}/{total} params "
                    f"(SoLEXS + HEL1OS branches)")
        logger.info("Trainable: Overlap Conv + BiLSTM + Multi-head outputs")

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.get("transfer_learning", {}).get("fine_tune_lr", 0.0001),
            weight_decay=self.cfg.get("weight_decay", 0.0001)
        )
