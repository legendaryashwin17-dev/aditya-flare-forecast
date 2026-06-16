#!/usr/bin/env python3
"""
Aditya-L1 Solar Flare Forecasting & Nowcasting — Main Entry Point.

PRIMARY DATA: Aditya-L1 SoLEXS + HEL1OS (Level-1, ISSDC PRADAN)
SUPPLEMENTARY: GOES XRS + HEK (training augmentation)

Bharatiya Antariksh Hackathon 2026 — Problem Statement 15

Usage:
    python main.py --mode pipeline                    # Full pipeline (simulated SoLEXS+HEL1OS)
    python main.py --mode pipeline --data-dir ./data  # Full pipeline from FITS files
    python main.py --mode train --solexs ./solexs.fits --hel1os ./hel1os.fits
    python main.py --mode evaluate --checkpoint data/models/best_model.pt
    python main.py --mode dashboard
    python main.py --mode nowcast --solexs ./solexs.fits --hel1os ./hel1os.fits
"""

import os
import sys
import argparse
import logging

from config import load_config
from src.utils import set_seed, setup_logging
from src.pipeline import FlarePipeline

logger = logging.getLogger(__name__)


def run_pipeline(args):
    """Run the end-to-end pipeline on Aditya-L1 SoLEXS + HEL1OS data."""
    pipeline = FlarePipeline()
    results = pipeline.run_all(
        solexs_paths=[args.solexs] if args.solexs else None,
        hel1os_paths=[args.hel1os] if args.hel1os else None,
        use_supplementary=not args.no_supplementary,
        train=True,
        evaluate=True
    )

    if results["nowcast_catalogue"] is not None and len(results["nowcast_catalogue"]) > 0:
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)
        cat_path = os.path.join(out_dir, "master_flare_catalogue.csv")
        results["nowcast_catalogue"].to_csv(cat_path, index=False)
        logger.info(f"Master catalogue saved to {cat_path}")

    if results["eval_metrics"]:
        logger.info(f"Test set metrics: {results['eval_metrics']['mean_tss_across_horizons']:.4f} TSS")

    return results


def run_training(args):
    """Train model on Aditya-L1 data."""
    pipeline = FlarePipeline()
    pipeline.run_data_ingestion(
        solexs_paths=[args.solexs] if args.solexs else None,
        hel1os_paths=[args.hel1os] if args.hel1os else None,
        use_supplementary=not args.no_supplementary
    )
    pipeline.run_preprocessing()
    pipeline.run_training()
    pipeline.run_evaluation()


def run_evaluation(args):
    """Evaluate a trained model checkpoint on Aditya-L1 test data."""
    pipeline = FlarePipeline()
    pipeline.run_data_ingestion(
        use_supplementary=not args.no_supplementary
    )
    pipeline.run_preprocessing()

    import torch
    from src.models import FlareForecastModel

    config = pipeline.config
    df = pipeline.processed_data
    feature_cols = [c for c in df.columns if c != "is_valid"]
    n_features = len(feature_cols)
    horizons = config["data"]["forecast_horizons_minutes"]

    model = FlareForecastModel(
        n_features=n_features, horizons=horizons,
        cnn_config=config["model"]["cnn"],
        lstm_config=config["model"]["lstm"],
        dense_config=config["model"]["dense"],
    )

    checkpoint_path = args.checkpoint or "data/models/best_model.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})")
    else:
        logger.warning(f"No checkpoint at {checkpoint_path} — using untrained model")

    from src.data.dataset import create_labels_from_flares, create_data_loaders
    features = df[feature_cols].values.astype(np.float32)
    labels = create_labels_from_flares(df, pipeline.flare_catalogue, horizons)
    _, _, test_loader, _ = create_data_loaders(features, df.index, labels, horizons, config)

    from src.train import Trainer
    trainer = Trainer(model, config)
    metrics = trainer.evaluate(test_loader)
    return metrics


def run_dashboard(args):
    """Launch the interactive Aditya-L1 monitoring dashboard.

    Shows:
        - SoLEXS + HEL1OS light curves (real-time)
        - Hardness ratio panel
        - Nowcast detection markers
        - Forecast probabilities for 15/30/60 min horizons
        - Visual alerts on flare detection
    """
    pipeline = FlarePipeline()
    logger.info("Launching dashboard at http://127.0.0.1:8050")
    logger.info("Open in browser to view SoLEXS+HEL1OS light curves with flare alerts")
    pipeline.run_dashboard()


def run_nowcast(args):
    """Run nowcasting on SoLEXS + HEL1OS FITS files, generate master catalogue."""
    pipeline = FlarePipeline()
    pipeline.run_data_ingestion(
        solexs_paths=[args.solexs] if args.solexs else None,
        hel1os_paths=[args.hel1os] if args.hel1os else None,
        use_supplementary=False
    )
    pipeline.run_preprocessing()
    merged = pipeline.run_nowcasting()

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    cat_path = os.path.join(out_dir, "nowcast_catalogue.csv")
    merged.to_csv(cat_path, index=False)
    logger.info(f"Nowcast catalogue saved to {cat_path}")
    logger.info(f"Total flare events detected: {len(merged)}")


def main():
    parser = argparse.ArgumentParser(
        description="Aditya-L1 Solar Flare Forecasting & Nowcasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--mode", type=str, default="pipeline",
        choices=["pipeline", "train", "evaluate", "dashboard", "nowcast"],
        help="Pipeline mode")
    parser.add_argument("--solexs", type=str, default=None,
        help="Path to SoLEXS Level-1 FITS file (PRIMARY)")
    parser.add_argument("--hel1os", type=str, default=None,
        help="Path to HEL1OS Level-1 FITS file (PRIMARY)")
    parser.add_argument("--data-dir", type=str, default=None,
        help="Directory containing SoLEXS+HEL1OS FITS files (auto-discovered)")
    parser.add_argument("--checkpoint", type=str, default=None,
        help="Path to trained model checkpoint")
    parser.add_argument("--output-dir", type=str, default="outputs/catalogues",
        help="Output directory for catalogues")
    parser.add_argument("--no-supplementary", action="store_true",
        help="Disable supplementary GOES XRS data")
    parser.add_argument("--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed")

    args = parser.parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)

    if args.data_dir:
        args.solexs = args.data_dir
        args.hel1os = args.data_dir

    mode_map = {
        "pipeline": run_pipeline,
        "train": run_training,
        "evaluate": run_evaluation,
        "dashboard": run_dashboard,
        "nowcast": run_nowcast,
    }

    logger.info(f"Aditya-L1 Flare Forecasting — Mode: {args.mode}")
    logger.info(f"Primary data source: SoLEXS (2-22 keV) + HEL1OS (8-150 keV)")

    mode_fn = mode_map.get(args.mode)
    if mode_fn:
        mode_fn(args)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
