"""
End-to-end pipeline: Aditya-L1 Solar Flare Forecasting & Nowcasting.

Orchestrates:
    1. PRIMARY: Aditya-L1 SoLEXS + HEL1OS data ingestion (ISSDC PRADAN)
    2. SUPPLEMENTARY: GOES XRS + HEK for training augmentation
    3. Preprocessing and physics-informed feature engineering
    4. Nowcasting (threshold-based detection on SoLEXS + HEL1OS)
    5. Forecasting (CNN-BiLSTM-Attention model)
    6. Master catalogue generation
    7. Evaluation and visualization

References:
    - Nandi et al. 2025, arXiv:2512.12679 (HEL1OS instrument)
    - Sarwade et al. 2025, arXiv:2509.26292 (SoLEXS instrument)
    - Hassani et al. 2025, ApJS, 279, 27 (LSTM/DLSTM benchmark TSS=0.74)
"""

import os
import logging
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import torch

from config import load_config
from src.data import (AdityaL1DataIngester, SupplementaryDataIngester,
                       FlarePreprocessor, read_solexs_and_hel1os_unified)
from src.data.dataset import (create_labels_from_flares,
                               create_data_loaders)
from src.models import ParallelFlareModel
from src.nowcasting import Nowcaster, FlareCatalogueMerger
from src.evaluation import FlareEvaluationMetrics
from src.visualization import LightCurvePlotter, MetricsPlotter, FlareDashboard
from src.train import Trainer
from src.utils import set_seed, setup_logging

logger = logging.getLogger(__name__)


class FlarePipeline:
    """End-to-end solar flare nowcasting and forecasting pipeline.

    PRIMARY: Aditya-L1 SoLEXS + HEL1OS
    SUPPLEMENTARY: GOES XRS + HEK (training augmentation)
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        set_seed(42)

        self.preprocessor = FlarePreprocessor(self.config)
        self.nowcaster = Nowcaster(self.config)
        self.catalogue_merger = FlareCatalogueMerger(
            self.config["nowcasting"]["merge_window_seconds"]
        )
        self.metrics_eval = FlareEvaluationMetrics(
            self.config["evaluation"]["thresholds"]
        )
        self.plotter = LightCurvePlotter()
        self.metrics_plotter = MetricsPlotter()

        self.raw_flux = None
        self.flare_catalogue = None
        self.processed_data = None
        self.model = None
        self.trainer = None
        self.nowcast_catalogue = None
        self.eval_metrics = None
        self.training_history = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.feature_params = None

        logger.info("FlarePipeline initialized (primary: Aditya-L1 SoLEXS+HEL1OS)")

    def run_data_ingestion(self, solexs_paths: List[str] = None,
                            hel1os_paths: List[str] = None,
                            use_supplementary: bool = True):
        """Stage 1: Ingest Aditya-L1 SoLEXS + HEL1OS (PRIMARY).

        Optionally ingests supplementary GOES XRS + HEK for training augmentation.
        """
        logger.info("=" * 60)
        logger.info("Stage 1: Data Ingestion")
        logger.info("PRIMARY: Aditya-L1 SoLEXS + HEL1OS")
        logger.info("=" * 60)

        aditya_cfg = self.config["data"].get("aditya_l1", {})
        aditya_start = aditya_cfg.get("aditya_l1_start_date", "2024-07-01")
        aditya_end = aditya_cfg.get("aditya_l1_end_date", "2025-12-31")
        aditya_ingester = AdityaL1DataIngester(
            start_date=aditya_start,
            end_date=aditya_end
        )

        if solexs_paths or hel1os_paths:
            logger.info("Loading Aditya-L1 from provided FITS paths")
            solexs, hel1os = aditya_ingester.load_aditya_l1_both(
                solexs_paths or [], hel1os_paths or []
            )
        else:
            data_dir = self.config["data"].get("aditya_l1_data_dir", "data/raw")
            if os.path.isdir(data_dir) and any(
                f.endswith(".fits") for f in os.listdir(data_dir)
            ):
                logger.info(f"Auto-discovering Aditya-L1 FITS in {data_dir}")
                solexs, hel1os = aditya_ingester.load_from_directory(data_dir)
            else:
                logger.info("No Aditya-L1 FITS found. Using simulated SoLEXS+HEL1OS for pipeline dev")
                solexs, hel1os = aditya_ingester.load_aditya_l1_both([], [])

        df = pd.DataFrame(index=solexs.index)
        df["solexs_flux"] = solexs["solexs_flux"]
        df = df.join(hel1os["hel1os_flux"], how="outer")
        df = df.ffill().dropna()

        self.raw_flux = df
        logger.info(f"Aditya-L1 combined dataset: {len(df)} samples")

        self.flare_catalogue = pd.DataFrame(
            columns=["start_time", "peak_time", "end_time", "goes_class", "peak_flux"]
        )

        if use_supplementary:
            try:
                self._ingest_supplementary(df)
            except Exception as e:
                logger.warning(f"Supplementary ingestion failed: {e}")

        return df

    def _ingest_supplementary(self, aditya_df: pd.DataFrame):
        """Ingest supplementary GOES XRS + HEK data for training augmentation."""
        logger.info("SUPPLEMENTARY: GOES XRS + HEK flare catalogue")
        supp_cfg = self.config["data"]["supplementary"]["goes"]
        if not supp_cfg.get("enabled", True):
            return

        supp_ingester = SupplementaryDataIngester(
            supp_cfg["start_date"], supp_cfg["end_date"],
            supp_cfg["cadence_seconds"]
        )
        goes_flux = supp_ingester.fetch_goes_xrs()
        hek_catalogue = supp_ingester.fetch_hek_flare_catalogue()

        goes_df = pd.DataFrame({
            "soft_flux": goes_flux["xrs_b_flux"],
            "hard_flux": goes_flux["xrs_a_flux"]
        }, index=goes_flux.index)

        self.supplementary_flux = goes_df
        self.supplementary_flare_catalogue = hek_catalogue

        logger.info(f"Supplementary GOES: {len(goes_df)} samples")
        logger.info(f"Supplementary HEK: {len(hek_catalogue)} flares")

    def run_preprocessing(self):
        """Stage 2: Preprocess and compute physics-informed features.

        Features:
            - Soft/hard flux (log-transformed)
            - Hardness ratio (HEL1OS/SoLEXS)
            - dF/dt, d²F/dt² (Neupert effect)
            - Rolling statistics (5-min, 15-min windows)
            - Background-subtracted flux
            - Cross-correlation lag
        """
        logger.info("=" * 60)
        logger.info("Stage 2: Preprocessing & Feature Engineering")
        logger.info("=" * 60)

        if self.raw_flux is None:
            raise RuntimeError("Run data ingestion first")

        df = self.raw_flux.copy().ffill().dropna()
        soft_df = df[["solexs_flux"]].copy()
        hard_df = df[["hel1os_flux"]].copy()
        df = self.preprocessor.unify_and_bin(soft_df, hard_df,
                                              target_cadence_s=10)
        df = self.preprocessor.handle_gaps(df)
        df = self.preprocessor.compute_features(df)
        df, self.feature_params = self.preprocessor.standardize(df)

        self.processed_data = df
        logger.info(f"Processed {len(df)} samples with {len(df.columns)} features")
        return df

    def run_nowcasting(self):
        """Stage 3: Real-time threshold-based nowcasting on SoLEXS + HEL1OS.

        Detection criteria:
            1. Flux > background + 3σ for 3 consecutive samples
            2. dF/dt > 2σ of rolling derivative
        Output: Master flare catalogue (explicit deliverable)
        """
        logger.info("=" * 60)
        logger.info("Stage 3: Nowcasting (Threshold Detection)")
        logger.info("=" * 60)

        if self.processed_data is None:
            raise RuntimeError("Run preprocessing first")

        df = self.processed_data
        times = df.index
        soft_flux = df["solexs_flux"].values
        hard_flux = df["hel1os_flux"].values

        detections = self.nowcaster.nowcast(times, soft_flux, hard_flux)
        merged = self.catalogue_merger.merge(
            [detections[detections["channel"] == "SoLEXS"],
             detections[detections["channel"] == "HEL1OS"]],
            ["SoLEXS", "HEL1OS"]
        )

        self.nowcast_catalogue = merged
        logger.info(f"Master nowcast catalogue: {len(merged)} events")
        if len(merged) > 0:
            logger.info(f"  Classes: {merged['goes_class'].value_counts().to_dict()}")
            logger.info(f"  Sources: {merged['sources'].value_counts().to_dict()}")
        return merged

    def run_training(self):
        """Stage 4: Train CNN-BiLSTM-Attention forecasting model.

        Uses Aditya-L1 SoLEXS+HEL1OS features as primary input.
        When supplementary GOES data is available, augments training.
        """
        logger.info("=" * 60)
        logger.info("Stage 4: Training Forecasting Model")
        logger.info("Architecture: ParallelConv1D(32,k=3) -> OverlapConv(64,k=5) -> BiLSTM(64) -> 3 Heads")
        logger.info("Loss: Focal Loss (alpha=0.75, gamma=2.0)")
        logger.info("Optimizer: Adam (lr=1e-3, wd=1e-4)")
        logger.info("=" * 60)

        if self.processed_data is None:
            raise RuntimeError("Run preprocessing first")

        df = self.processed_data
        feature_cols = [c for c in df.columns if c != "is_valid"]
        features = df[feature_cols].values.astype(np.float32)
        timestamps = df.index

        horizons = self.config["data"]["forecast_horizons_minutes"]

        flare_cat = self.flare_catalogue
        if hasattr(self, "supplementary_flare_catalogue") and len(self.supplementary_flare_catalogue) > 0:
            logger.info(f"Augmenting labels with {len(self.supplementary_flare_catalogue)} supplementary flares")

        labels = create_labels_from_flares(df, flare_cat, horizons)

        train_loader, val_loader, test_loader, pos_weight = \
            create_data_loaders(features, timestamps, labels, horizons, self.config)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        sample_x, sample_y = next(iter(train_loader))
        n_features = sample_x.shape[2]

        soft_channels = self.config.get("features", {}).get("soft_channels", ["solexs_flux"])
        n_solexs = len(soft_channels)
        n_hel1os = n_features - n_solexs

        model = ParallelFlareModel(
            n_solexs_features=n_solexs,
            n_hel1os_features=n_hel1os,
            config=self.config
        )

        self.model = model
        self.trainer = Trainer(model, self.config)

        history = self.trainer.fit(
            train_loader, val_loader,
            checkpoint_dir=self.config.get("checkpoint_dir", "data/models")
        )

        self.training_history = history
        logger.info(f"Training complete. Best val TSS: {history['best_val_tss']:.4f}")
        return history

    def run_evaluation(self):
        """Stage 5: Evaluate on held-out test set."""
        logger.info("=" * 60)
        logger.info("Stage 5: Evaluation")
        logger.info("Metrics: TSS, HSS, FAR, POD, AUC, Lead Time")
        logger.info("=" * 60)

        if self.trainer is None:
            raise RuntimeError("Run training first")

        metrics = self.trainer.evaluate(self.test_loader)
        self.eval_metrics = metrics
        return metrics

    def run_all(self, solexs_paths: List[str] = None,
                hel1os_paths: List[str] = None,
                use_supplementary: bool = True,
                train: bool = True,
                evaluate: bool = True) -> Dict:
        """Run the full end-to-end pipeline."""
        logger.info("=" * 60)
        logger.info("ADITYA-L1 SOLAR FLARE FORECASTING PIPELINE")
        logger.info("Primary: SoLEXS (2-22 keV) + HEL1OS (8-150 keV)")
        logger.info("=" * 60)

        self.run_data_ingestion(solexs_paths, hel1os_paths, use_supplementary)
        self.run_preprocessing()
        self.run_nowcasting()

        if train:
            self.run_training()
            if evaluate:
                self.run_evaluation()

        logger.info("Pipeline complete!")
        return {
            "nowcast_catalogue": self.nowcast_catalogue,
            "eval_metrics": self.eval_metrics,
            "training_history": self.training_history
        }
