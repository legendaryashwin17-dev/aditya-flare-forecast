"""
Training script for Aditya-L1 Solar Flare Forecasting.

Supports:
    - Training on real PRADAN data (SoLEXS + HEL1OS)
    - Transfer learning from GOES XRS pre-training
    - Focal Loss for class imbalance
    - Multi-horizon forecasting (15/30/60 min)

Usage:
    python -m src.train_pradan --mode=train_real
    python -m src.train_pradan --mode=transfer_learn --goes_checkpoint=data/models/goes_pretrained.pt
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models import ParallelFlareModel, FocalLoss
from src.data.preprocessing import FlarePreprocessor
from src.data.dataset import FlareWindowDataset, create_labels_from_flares, create_data_loaders
from src.data.goes_loader import fetch_goes_xrs, create_goes_flare_catalogue, load_goes_from_file
from src.evaluation import FlareEvaluationMetrics
from src.app.utils.fits_reader import read_solexs_lc, read_hel1os_lc, find_fits_in_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load configuration from config.yaml."""
    import yaml
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_real_pradan_data(
    solexs_dir: str = "data/pradan_solexs",
    hel1os_dir: str = "data/pradan_hel1os",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load real PRADAN data from extracted directories."""
    logger.info("Loading real PRADAN data...")

    solexs_path, _ = find_fits_in_dir(solexs_dir)
    if solexs_path:
        solexs_df = read_solexs_lc(solexs_path)
        logger.info(f"  SoLEXS: {len(solexs_df)} samples")
    else:
        logger.warning("  No SoLEXS data found")
        solexs_df = None

    _, hel1os_path = find_fits_in_dir(hel1os_dir)
    if hel1os_path:
        hel1os_df = read_hel1os_lc(hel1os_path, band="full")
        logger.info(f"  HEL1OS: {len(hel1os_df)} samples")
    else:
        logger.warning("  No HEL1OS data found")
        hel1os_df = None

    return solexs_df, hel1os_df


def get_flare_labels_from_goes(
    goes_catalogue: pd.DataFrame,
    aditya_times: pd.DatetimeIndex,
    horizons_minutes: List[int],
) -> dict:
    """Create flare labels from GOES catalogue for Aditya-L1 timestamps.

    Maps GOES flare times to Aditya-L1 time grid.
    """
    labels = {h: np.zeros(len(aditya_times), dtype=np.float32)
              for h in horizons_minutes}

    if goes_catalogue.empty:
        logger.warning("Empty GOES catalogue - no flare labels")
        return labels

    times = aditya_times

    for _, flare in goes_catalogue.iterrows():
        peak_time = pd.Timestamp(flare["peak_time"])

        for h in horizons_minutes:
            # Label timesteps within [peak - h minutes, peak] as positive
            start_time = peak_time - pd.Timedelta(minutes=h)
            mask = (times >= start_time) & (times <= peak_time)
            labels[h][mask] = 1.0

    for h in horizons_minutes:
        n_pos = int(labels[h].sum())
        logger.info(f"  Horizon {h}min: {n_pos} positive / {len(labels[h])} total "
                    f"({n_pos/len(labels[h])*100:.2f}%)")

    return labels


def prepare_training_data(
    config: dict,
    use_real_data: bool = True,
    goes_catalogue_path: Optional[str] = None,
) -> Tuple[np.ndarray, pd.DatetimeIndex, dict, dict]:
    """Prepare training data from real PRADAN with GOES flare labels."""
    preprocessor = FlarePreprocessor(config)

    if use_real_data:
        solexs_df, hel1os_df = load_real_pradan_data()
        if solexs_df is None and hel1os_df is None:
            logger.error("No real data available")
            raise ValueError("No PRADAN data found")
    else:
        from src.app.utils.simulated import generate_simulated_data
        solexs_df, hel1os_df, _ = generate_simulated_data(
            duration_hours=24.0, n_flares=10, seed=42)

    # Preprocess
    combined = preprocessor.unify_and_bin(solexs_df, hel1os_df, target_cadence_s=10)
    combined = preprocessor.handle_gaps(combined)
    combined = preprocessor.compute_features(combined)
    combined, feature_params = preprocessor.standardize(combined)

    feature_cols = [c for c in combined.columns if c != "is_valid"]
    features = combined[feature_cols].values.astype(np.float32)
    timestamps = combined.index

    # Get flare labels
    horizons = config["data"]["forecast_horizons_minutes"]

    if goes_catalogue_path and os.path.exists(goes_catalogue_path):
        # Use real GOES flare catalogue
        logger.info(f"Loading GOES catalogue from: {goes_catalogue_path}")
        goes_cat = pd.read_csv(goes_catalogue_path, parse_dates=["peak_time"])
        labels = get_flare_labels_from_goes(goes_cat, timestamps, horizons)
    elif use_real_data:
        # Try to fetch GOES data for flare labels
        logger.info("Fetching GOES data for flare labels...")
        try:
            goes_df = fetch_goes_xrs("2024-01-01", "2024-12-31")
            if not goes_df.empty:
                goes_cat = create_goes_flare_catalogue(goes_df, min_class="C")
                labels = get_flare_labels_from_goes(goes_cat, timestamps, horizons)
            else:
                logger.warning("No GOES data - using synthetic labels")
                from src.app.utils.simulated import generate_simulated_data
                _, _, flare_cat = generate_simulated_data(
                    duration_hours=24.0, n_flares=10, seed=42)
                labels = create_labels_from_flares(
                    pd.DataFrame(index=timestamps), flare_cat, horizons)
        except Exception as e:
            logger.warning(f"Failed to fetch GOES: {e}")
            from src.app.utils.simulated import generate_simulated_data
            _, _, flare_cat = generate_simulated_data(
                duration_hours=24.0, n_flares=10, seed=42)
            labels = create_labels_from_flares(
                pd.DataFrame(index=timestamps), flare_cat, horizons)
    else:
        from src.app.utils.simulated import generate_simulated_data
        _, _, flare_cat = generate_simulated_data(
            duration_hours=24.0, n_flares=10, seed=42)
        labels = create_labels_from_flares(
            pd.DataFrame(index=timestamps), flare_cat, horizons)

    logger.info(f"Training data: {features.shape[0]} samples, {features.shape[1]} features")

    return features, timestamps, labels, {"feature_params": feature_params}


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
                logger.info(f"  {h}: TSS={tss:.4f}, AUC={auc:.4f}")

        logger.info(f"  Mean TSS: {metrics.get('mean_tss_across_horizons', 0):.4f}")
        logger.info(f"  Mean AUC: {metrics.get('mean_auc_across_horizons', 0):.4f}")
        return metrics

    def freeze_branches_for_finetune(self):
        """Freeze SoLEXS and HEL1OS branch CNNs for transfer learning."""
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


def train_on_real_data(config: dict, checkpoint_dir: str = "data/models",
                       goes_catalogue_path: Optional[str] = None):
    """Train model on real PRADAN data with GOES flare labels."""
    logger.info("=" * 60)
    logger.info("TRAINING ON REAL PRADAN DATA")
    logger.info("=" * 60)

    features, timestamps, labels, meta = prepare_training_data(
        config, use_real_data=True, goes_catalogue_path=goes_catalogue_path)

    horizons = config["data"]["forecast_horizons_minutes"]
    train_loader, val_loader, test_loader, pos_weight = create_data_loaders(
        features, timestamps, labels, horizons, config)

    logger.info(f"Pos weight: {pos_weight:.2f}")
    logger.info(f"Train: {len(train_loader.dataset)} windows")
    logger.info(f"Val: {len(val_loader.dataset)} windows")
    logger.info(f"Test: {len(test_loader.dataset)} windows")

    n_features = features.shape[1]
    n_solexs = len(config["features"].get("soft_channels", []))
    n_hel1os = n_features - n_solexs

    model = ParallelFlareModel(n_solexs, n_hel1os, config)
    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    trainer = Trainer(model, config)
    results = trainer.fit(train_loader, val_loader,
                          max_epochs=config["training"]["max_epochs"],
                          checkpoint_dir=checkpoint_dir)

    test_metrics = trainer.evaluate(test_loader)

    return results, test_metrics


def transfer_learn_from_goes(config: dict, goes_checkpoint: str,
                             checkpoint_dir: str = "data/models",
                             goes_catalogue_path: Optional[str] = None):
    """Transfer learn from GOES pre-trained model to Aditya-L1."""
    logger.info("=" * 60)
    logger.info("TRANSFER LEARNING: GOES -> Aditya-L1")
    logger.info("=" * 60)

    if not os.path.exists(goes_checkpoint):
        logger.error(f"GOES checkpoint not found: {goes_checkpoint}")
        logger.info("Train on GOES first: python -m src.train_pradan --mode=train_goes")
        return None

    checkpoint = torch.load(goes_checkpoint, map_location="cpu")
    goes_config = checkpoint.get("config", config)

    # Load GOES pre-trained model
    goes_model = ParallelFlareModel(
        n_solexs_features=2,
        n_hel1os_features=1,
        config=goes_config
    )
    goes_model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded GOES model from epoch {checkpoint.get('epoch', '?')}")

    # Prepare Aditya-L1 data
    features, timestamps, labels, meta = prepare_training_data(
        config, use_real_data=True, goes_catalogue_path=goes_catalogue_path)

    horizons = config["data"]["forecast_horizons_minutes"]
    train_loader, val_loader, test_loader, pos_weight = create_data_loaders(
        features, timestamps, labels, horizons, config)

    # Create Aditya-L1 model
    n_features = features.shape[1]
    n_solexs = len(config["features"].get("soft_channels", []))
    n_hel1os = n_features - n_solexs

    aditya_model = ParallelFlareModel(n_solexs, n_hel1os, config)

    # Transfer compatible weights (branches that match dimensions)
    goes_state = goes_model.state_dict()
    aditya_state = aditya_model.state_dict()

    transferred = 0
    for key in aditya_state:
        if key in goes_state and goes_state[key].shape == aditya_state[key].shape:
            aditya_state[key] = goes_state[key]
            transferred += 1

    aditya_model.load_state_dict(aditya_state)
    logger.info(f"Transferred {transferred} weight tensors from GOES model")

    # Train with frozen branches
    trainer = Trainer(aditya_model, config)
    trainer.freeze_branches_for_finetune()

    results = trainer.fit(train_loader, val_loader,
                          max_epochs=config["training"]["max_epochs"],
                          checkpoint_dir=checkpoint_dir)

    test_metrics = trainer.evaluate(test_loader)

    return results, test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Aditya-L1 flare forecasting model")
    parser.add_argument("--mode",
                        choices=["train_real", "transfer_learn", "train_goes"],
                        default="train_real",
                        help="Training mode")
    parser.add_argument("--goes_checkpoint", type=str, default=None,
                        help="Path to GOES pre-trained checkpoint")
    parser.add_argument("--goes_catalogue", type=str, default=None,
                        help="Path to GOES flare catalogue CSV")
    parser.add_argument("--checkpoint_dir", type=str, default="data/models",
                        help="Directory to save model checkpoints")
    args = parser.parse_args()

    config = load_config()

    if args.mode == "train_real":
        results, metrics = train_on_real_data(
            config, args.checkpoint_dir, args.goes_catalogue)
    elif args.mode == "transfer_learn":
        if args.goes_checkpoint is None:
            logger.error("Please provide --goes_checkpoint for transfer learning")
            sys.exit(1)
        results, metrics = transfer_learn_from_goes(
            config, args.goes_checkpoint, args.checkpoint_dir,
            args.goes_catalogue)
    elif args.mode == "train_goes":
        logger.info("Training on GOES data for transfer learning...")
        # TODO: Implement GOES-only pre-training
        logger.error("GOES pre-training not yet implemented")
        sys.exit(1)

    if results is not None:
        import json
        results_path = Path(args.checkpoint_dir).parent / "training_results.json"

        # Serialize metrics (remove non-serializable items)
        serializable_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float, str, bool)):
                serializable_metrics[k] = v
            elif isinstance(v, dict):
                serializable_metrics[k] = {
                    mk: mv for mk, mv in v.items()
                    if isinstance(mv, (int, float, str, bool))
                }

        with open(results_path, "w") as f:
            json.dump({
                "mode": args.mode,
                "best_val_tss": results["best_val_tss"],
                "best_epoch": results["best_epoch"],
                "test_metrics": serializable_metrics
            }, f, indent=2)

        logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
