"""
End-to-end training demo with realistic flare-injected data.

Injects synthetic flare profiles into simulated SoLEXS+HEL1OS light curves
using physically motivated profiles (fast rise, exponential decay),
then trains the ParallelFlareModel and reports TSS/AUC per horizon.
"""

import os, sys, logging, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import load_config
from src.data.preprocessing import FlarePreprocessor, merge_soft_hard
from src.data.dataset import FlareWindowDataset, create_labels_from_flares, create_data_loaders
from src.models import ParallelFlareModel, FocalLoss
from src.train import Trainer
from src.evaluation import FlareEvaluationMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

def inject_flare_profiles(n_points, n_flares=50, seed=42):
    """Generate SoLEXS+HEL1OS data with injected realistic flare profiles.

    Flare shape: fast-rise (1-3 min) + exponential decay (5-15 min)
    Classes: A through X scaled by peak flux
    """
    rng = np.random.RandomState(seed)
    cadence = 10  # seconds
    total_seconds = n_points * cadence
    times = pd.date_range("2024-07-01", periods=n_points, freq=f"{cadence}s")

    quiet_soft = 1e-6
    quiet_hard = 3e-7

    soft_flux = quiet_soft * (1 + 0.05 * rng.randn(n_points))
    hard_flux = quiet_hard * (1 + 0.08 * rng.randn(n_points))

    flare_catalogue = []
    flare_classes = ["B", "C", "M", "X"]
    class_peak_fluxes = {
        "B": (1e-7, 1e-6),
        "C": (1e-6, 1e-5),
        "M": (1e-5, 1e-4),
        "X": (1e-4, 2e-4),
    }
    class_weights = {"B": 0.3, "C": 0.4, "M": 0.2, "X": 0.1}

    for _ in range(n_flares):
        cls = rng.choice(flare_classes, p=[class_weights[c] for c in flare_classes])
        peak_min, peak_max = class_peak_fluxes[cls]
        peak_flux = rng.uniform(peak_min, peak_max)

        # Flare timing
        peak_idx = rng.randint(200, n_points - 300)
        rise_duration = int(rng.uniform(6, 18))        # 1-3 min at 10s
        decay_duration = int(rng.uniform(30, 90))       # 5-15 min
        precursor_duration = int(rng.uniform(90, 150))  # 15-25 min precursor

        # --- Precursor phase: gradual HEL1OS brightening 10-20 min before peak ---
        # Based on Nandi et al. 2025: pre-flare brightening near 20 keV
        flare_start = peak_idx - rise_duration
        precursor_start = flare_start - precursor_duration
        if precursor_start >= 0:
            t_pre = np.arange(precursor_duration)
            precursor_rise = 1.0 - np.exp(-t_pre / (precursor_duration / 4))
            precursor_amp = rng.uniform(0.15, 0.40) * peak_flux
            hard_flux[precursor_start:flare_start] += precursor_rise * precursor_amp
            soft_flux[precursor_start:flare_start] += precursor_rise * precursor_amp * rng.uniform(0.1, 0.25)

        # --- Main flare: fast rise + exponential decay ---
        rise = np.exp(-np.linspace(3, 0, rise_duration)**2 / 4.5)
        t_decay = np.arange(decay_duration)
        decay = np.exp(-t_decay / (decay_duration / 3.5))
        profile = np.concatenate([rise, decay])
        profile = profile / profile.max() * peak_flux

        start = flare_start
        end = min(start + len(profile), n_points)
        actual_len = end - start

        soft_flux[start:end] += profile[:actual_len]
        # HEL1OS peaks higher and decays faster (Neupert effect)
        hard_profile = profile[:actual_len] * rng.uniform(0.3, 0.8)
        hard_profile[:rise_duration] *= rng.uniform(1.5, 3.0)
        hard_flux[start:end] += hard_profile

        flare_catalogue.append({
            "peak_time": times[peak_idx],
            "start_time": times[max(precursor_start, 0)],
            "end_time": times[min(n_points - 1, end)],
            "goes_class": cls,
            "peak_flux": peak_flux,
        })

    soft_flux = np.maximum(soft_flux, 1e-10)
    hard_flux = np.maximum(hard_flux, 1e-10)

    df = pd.DataFrame({
        "solexs_flux": soft_flux,
        "hel1os_flux": hard_flux,
    }, index=times)

    flare_df = pd.DataFrame(flare_catalogue)
    logger.info(f"Generated {len(flare_df)} flares:")
    logger.info(f"  Classes: {flare_df['goes_class'].value_counts().sort_index().to_dict()}")
    logger.info(f"  Peak flux range: {flare_df['peak_flux'].min():.2e} - {flare_df['peak_flux'].max():.2e}")

    return df, flare_df

def train_and_evaluate(config_path=None):
    logger.info("=" * 60)
    logger.info("ADITYA-L1 FLARE FORECAST - TRAINING DEMO WITH INJECTED FLARES")
    logger.info("=" * 60)

    config = load_config(config_path)
    logger.info(f"Config loaded from: {config_path or 'default'}")

    # Generate data with injected flares
    logger.info("\n--- Generating data with injected flare profiles ---")
    n_points = 100000
    df_raw, flare_cat = inject_flare_profiles(n_points, n_flares=30, seed=42)
    total_hours = len(df_raw) * 10 / 3600
    logger.info(f"Raw data: {len(df_raw)} samples at 10s cadence ({total_hours:.1f} hours = {total_hours/24:.1f} days)")

    # Preprocess
    logger.info("\n--- Preprocessing ---")
    preproc = FlarePreprocessor(config)
    df_processed = preproc.handle_gaps(df_raw)
    df_processed = preproc.compute_features(df_processed)
    df_processed, feat_params = preproc.standardize(df_processed)
    logger.info(f"Processed: {len(df_processed)} samples x {len(df_processed.columns)} features")

    # Create labels
    logger.info("\n--- Creating labels ---")
    horizons = config["data"]["forecast_horizons_minutes"]
    labels = create_labels_from_flares(df_processed, flare_cat, horizons)

    # Create data loaders
    logger.info("\n--- Creating data loaders ---")
    feature_cols = [c for c in df_processed.columns if c != "is_valid"]
    features = df_processed[feature_cols].values.astype(np.float32)
    timestamps = df_processed.index

    train_loader, val_loader, test_loader, pos_weight = create_data_loaders(
        features, timestamps, labels, horizons, config
    )

    # Diagnostic: count positive samples in each split
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        n_pos = 0
        n_total = 0
        for _, y in loader:
            n_pos += y.sum().item()
            n_total += y.numel()
        logger.info(f"  {split_name}: {int(n_pos)} positive / {n_total} labels ({n_pos/n_total*100:.1f}%)")

    # Count positives in train
    sample_x, sample_y = next(iter(train_loader))
    n_features = sample_x.shape[2]
    logger.info(f"Feature dimension: {n_features}")
    logger.info(f"Pos weight from sampler: {pos_weight:.2f}")

    # Compute actual pos fraction
    all_y = []
    for _, y in train_loader:
        all_y.append(y)
    train_y = torch.cat(all_y)
    pos_frac = train_y.mean().item()
    logger.info(f"Train set positive fraction: {pos_frac:.4f}")

    # Create model
    logger.info("\n--- Creating ParallelFlareModel ---")
    soft_channels = config.get("features", {}).get("soft_channels", ["solexs_flux"])
    n_solexs = len(soft_channels)
    n_hel1os = n_features - n_solexs
    logger.info(f"Feature split: {n_solexs} SoLEXS + {n_hel1os} HEL1OS = {n_features} total")

    model = ParallelFlareModel(n_solexs_features=n_solexs, n_hel1os_features=n_hel1os, config=config)

    # Determine focal_alpha from actual pos fraction
    focal_alpha = max(0.5, min(0.95, 1.0 - pos_frac))
    config["training"]["focal_alpha"] = focal_alpha
    logger.info(f"Adjusted focal_alpha: {focal_alpha:.3f} (pos_frac={pos_frac:.4f})")

    # Train
    logger.info("\n--- Training ---")
    trainer = Trainer(model, config)
    history = trainer.fit(train_loader, val_loader, checkpoint_dir="data/models")

    # Evaluate
    logger.info("\n--- Test Set Evaluation ---")
    metrics = trainer.evaluate(test_loader)

    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"Best val TSS:     {history['best_val_tss']:.4f}")
    logger.info(f"Test mean TSS:    {metrics.get('mean_tss_across_horizons', 0):.4f}")
    logger.info(f"Test mean AUC:    {metrics.get('mean_auc_across_horizons', 0):.4f}")
    logger.info(f"Total params:     {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Epochs trained:   {len(history['train_losses'])}")

    for h_name in [f"{h}min" for h in horizons]:
        if h_name in metrics:
            m = metrics[h_name]
            best_pod = max(v for k, v in m.items() if k.startswith("pod@"))
            best_far = min(v for k, v in m.items() if k.startswith("far@"))
            logger.info(f"  {h_name}: TSS={m.get('best_tss', 0):.4f} "
                       f"AUC={m.get('auc', 0):.4f} "
                       f"POD={best_pod:.4f} "
                       f"FAR={best_far:.4f}")

    # Save metrics
    results = {
        "best_val_tss": float(history["best_val_tss"]),
        "test_mean_tss": float(metrics.get("mean_tss_across_horizons", 0)),
        "test_mean_auc": float(metrics.get("mean_auc_across_horizons", 0)),
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_flares": len(flare_cat),
        "pos_frac_train": float(pos_frac),
        "focal_alpha": float(focal_alpha),
        "per_horizon": {},
    }
    for h_name in [f"{h}min" for h in horizons]:
        if h_name in metrics:
            results["per_horizon"][h_name] = {
                k: float(v) if not isinstance(v, (dict, list)) else str(v)
                for k, v in metrics[h_name].items()
                if isinstance(v, (int, float))
            }

    os.makedirs("data", exist_ok=True)
    with open("data/training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to data/training_results.json")

    return history, metrics

if __name__ == "__main__":
    train_and_evaluate()
