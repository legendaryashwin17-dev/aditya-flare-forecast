"""
Evaluation metrics for solar flare forecasting.

Implements the standard space-weather forecasting metrics required by
the problem statement:

    Required:
        - True Positive Rate (POD)
        - False Alarm Rate (FAR)
        - Lead Time (minutes before peak)

    Standard in literature:
        - TSS (True Skill Statistic) — imbalance-invariant
        - HSS (Heidke Skill Score)  — skill vs random
        - AUC-ROC
        - BSS (Brier Skill Score)

Based on:
    - Problem Statement 15: Bharatiya Antariksh Hackathon 2026
    - Bloomfield et al. 2012, ApJ, 747, 41 (TSS/HSS standards)
    - Hassani et al. 2025, ApJS, 279, 27 (TSS=0.74 benchmark)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

logger = logging.getLogger(__name__)


class FlareEvaluationMetrics:
    """Compute all flare forecasting evaluation metrics."""

    def __init__(self, thresholds: List[float] = None):
        if thresholds is None:
            self.thresholds = [0.3, 0.5, 0.7]
        else:
            self.thresholds = thresholds

    def compute_all(self, y_true: np.ndarray, y_pred: np.ndarray,
                    horizons: List[int],
                    lead_times: Optional[np.ndarray] = None) -> Dict:
        """Compute full set of evaluation metrics.

        Args:
            y_true: [N, H] binary ground truth
            y_pred: [N, H] predicted probabilities
            horizons: list of forecast horizon names (minutes)
            lead_times: optional [N] array of lead times (NaN for non-flare windows)
        Returns:
            dict of metric_name -> value or metric_name -> dict of horizon->value
        """
        results = {}

        for h_idx, h in enumerate(horizons):
            yt = y_true[:, h_idx]
            yp = y_pred[:, h_idx]
            horizon_key = h if str(h).endswith("min") else f"{h}min"

            best_tss = -1.0
            best_res = None
            for thresh in self.thresholds:
                res = self._compute_binary_metrics(yt, yp, thresh)
                if res["tss"] > best_tss:
                    best_tss = res["tss"]
                    best_res = res
                if horizon_key not in results:
                    results[horizon_key] = {}
                for k, v in res.items():
                    results[horizon_key][f"{k}@{thresh}"] = v

            results[horizon_key]["best_tss"] = best_tss
            results[horizon_key]["auc"] = self._compute_auc(yt, yp)
            results[horizon_key]["brier"] = self._compute_brier(yt, yp)

            if lead_times is not None:
                mask = yt == 1
                if mask.sum() > 0:
                    lt = lead_times[mask]
                    results[horizon_key]["mean_lead_time"] = float(np.nanmean(lt))
                    results[horizon_key]["median_lead_time"] = float(np.nanmedian(lt))
                    results[horizon_key]["min_lead_time"] = float(np.nanmin(lt))
                    results[horizon_key]["max_lead_time"] = float(np.nanmax(lt))

        horizons_sorted = sorted(results.keys(), key=lambda k: int(k.replace("min", "")))
        avg_tss = np.mean([results[h]["best_tss"] for h in horizons_sorted])
        results["mean_tss_across_horizons"] = float(avg_tss)

        avg_auc = np.mean([results[h]["auc"] for h in horizons_sorted])
        results["mean_auc_across_horizons"] = float(avg_auc)

        self._log_summary(results)

        return results

    def _compute_binary_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                threshold: float) -> Dict[str, float]:
        """Compute binary classification metrics at a given threshold."""
        y_bin = (y_pred >= threshold).astype(np.float32)

        tp = np.sum((y_true == 1) & (y_bin == 1))
        tn = np.sum((y_true == 0) & (y_bin == 0))
        fp = np.sum((y_true == 0) & (y_bin == 1))
        fn = np.sum((y_true == 1) & (y_bin == 0))

        pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        pofd = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        tss = pod - pofd

        expected_correct = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / \
                           (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        total_correct = tp + tn
        total = tp + tn + fp + fn
        hss = (total_correct - expected_correct) / \
              (total - expected_correct) if total != expected_correct else 0.0

        return {
            "tss": float(tss),
            "hss": float(hss),
            "pod": float(pod),
            "pofd": float(pofd),
            "far": float(far),
            "precision": float(precision),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn)
        }

    def _compute_auc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(np.unique(y_true)) < 2:
            return 0.5
        try:
            return float(roc_auc_score(y_true, y_pred))
        except Exception:
            return 0.5

    def _compute_brier(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            return float(brier_score_loss(y_true, y_pred))
        except Exception:
            return 1.0

    def compute_tss(self, y_true: np.ndarray, y_pred: np.ndarray,
                    threshold: float = 0.5) -> float:
        res = self._compute_binary_metrics(y_true, y_pred, threshold)
        return res["tss"]

    def compute_far(self, y_true: np.ndarray, y_pred: np.ndarray,
                    threshold: float = 0.5) -> float:
        res = self._compute_binary_metrics(y_true, y_pred, threshold)
        return res["far"]

    def compute_lead_time(self, forecast_trigger_times: np.ndarray,
                          flare_peak_times: np.ndarray) -> Dict[str, float]:
        """Compute lead time statistics between forecast triggers and flare peaks."""
        if len(forecast_trigger_times) == 0 or len(flare_peak_times) == 0:
            return {"mean_lead_time": 0.0, "median_lead_time": 0.0}

        lead_times = []
        for ft in forecast_trigger_times:
            future_flares = flare_peak_times[flare_peak_times >= ft]
            if len(future_flares) > 0:
                lt = (future_flares[0] - ft) / np.timedelta64(1, 's') / 60.0
                lead_times.append(lt)

        if len(lead_times) == 0:
            return {"mean_lead_time": 0.0, "median_lead_time": 0.0}

        lt_arr = np.array(lead_times)
        return {
            "mean_lead_time": float(np.mean(lt_arr)),
            "median_lead_time": float(np.median(lt_arr)),
            "min_lead_time": float(np.min(lt_arr)),
            "max_lead_time": float(np.max(lt_arr)),
            "n_triggers": len(lead_times)
        }

    def _log_summary(self, results: Dict):
        horizons = sorted(
            [k for k in results if "min" in k],
            key=lambda x: int(x.replace("min", ""))
        )
        logger.info("=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        for h in horizons:
            logger.info(f"  Horizon {h}:")
            logger.info(f"    Best TSS:  {results[h]['best_tss']:.4f}")
            logger.info(f"    AUC:       {results[h]['auc']:.4f}")
            logger.info(f"    Brier:     {results[h]['brier']:.4f}")
            if "mean_lead_time" in results[h]:
                logger.info(f"    Mean lead time: {results[h]['mean_lead_time']:.1f} min")
                logger.info(f"    Median lead time: {results[h]['median_lead_time']:.1f} min")
        logger.info(f"  Mean TSS across horizons: {results['mean_tss_across_horizons']:.4f}")
        logger.info(f"  Mean AUC across horizons: {results['mean_auc_across_horizons']:.4f}")
        logger.info("=" * 60)
