"""
Static visualization plots for solar flare analysis.

Generates publication-quality figures of:
    - X-ray light curves with flare markers
    - Hardness ratio evolution
    - Neupert effect comparison (dSXR/dt vs HXR)
    - Model performance metrics (TSS, AUC, lead time curves)
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logger = logging.getLogger(__name__)


class LightCurvePlotter:
    """Plot X-ray light curves with detection and forecast markers."""

    @staticmethod
    def plot_light_curves(times: pd.DatetimeIndex, soft_flux: np.ndarray,
                          hard_flux: np.ndarray,
                          detections: Optional[pd.DataFrame] = None,
                          forecasts: Optional[pd.DataFrame] = None,
                          title: str = "SoLEXS + HEL1OS X-ray Light Curves",
                          save_path: Optional[str] = None):
        """Plot combined soft and hard X-ray light curves with event markers."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10),
                                             sharex=True)

        ax1.plot(times, soft_flux, color="blue", alpha=0.8, linewidth=0.8,
                 label="SoLEXS (Soft X-ray)")
        ax1.set_yscale("log")
        ax1.set_ylabel("Soft X-ray Flux", fontsize=11)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(title, fontsize=13, fontweight="bold")

        ax2.plot(times, hard_flux, color="red", alpha=0.8, linewidth=0.8,
                 label="HEL1OS (Hard X-ray)")
        ax2.set_yscale("log")
        ax2.set_ylabel("Hard X-ray Flux", fontsize=11)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        hardness = np.where(soft_flux > 1e-12,
                            hard_flux / soft_flux, 0.0)
        hardness_smooth = pd.Series(hardness).rolling(
            5, center=True, min_periods=1).mean()
        ax3.plot(times, hardness_smooth, color="green", alpha=0.8,
                 linewidth=0.8, label="Hardness Ratio (HEL1OS/SoLEXS)")
        ax3.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax3.set_ylabel("Hardness Ratio", fontsize=11)
        ax3.set_xlabel("Time (UTC)", fontsize=11)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)

        if detections is not None and len(detections) > 0:
            for _, event in detections.iterrows():
                for ax in [ax1, ax2]:
                    ax.axvline(x=event["peak_time"], color="orange",
                               linestyle="--", alpha=0.8, linewidth=1.5)
                    ax.axvspan(event["start_time"], event["end_time"],
                               alpha=0.15, color="orange")
            ax1.text(0.02, 0.95, f"{len(detections)} flares detected",
                     transform=ax1.transAxes, fontsize=10,
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        if forecasts is not None and len(forecasts) > 0:
            for _, fc in forecasts.iterrows():
                for ax in [ax1, ax2]:
                    ax.axvline(x=fc["trigger_time"], color="purple",
                               linestyle=":", alpha=0.8, linewidth=1.5)

        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        fig.autofmt_xdate()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved light curve plot to {save_path}")
        plt.close(fig)
        return fig

    @staticmethod
    def plot_neupert_effect(times: pd.DatetimeIndex, soft_flux: np.ndarray,
                            hard_flux: np.ndarray,
                            save_path: Optional[str] = None):
        """Plot Neupert effect: d(soft)/dt vs hard X-ray flux."""
        dsoft = np.gradient(soft_flux)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(dsoft, hard_flux, c=np.arange(len(dsoft)),
                   cmap="viridis", s=2, alpha=0.6)
        ax.set_xlabel("d(SoLEXS Flux)/dt", fontsize=12)
        ax.set_ylabel("HEL1OS Hard X-ray Flux", fontsize=12)
        ax.set_title("Neupert Effect: d(SXR)/dt vs HXR", fontsize=13,
                     fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        cbar = plt.colorbar(ax.collections[0])
        cbar.set_label("Time index", fontsize=10)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig


class MetricsPlotter:
    """Plot model evaluation metrics."""

    @staticmethod
    def plot_metrics_summary(metrics: Dict, save_path: Optional[str] = None):
        """Plot TSS, AUC, lead time for each forecast horizon."""
        horizons = sorted(
            [k for k in metrics if "min" in k],
            key=lambda x: int(x.replace("min", ""))
        )
        h_labels = [h.replace("min", "") for h in horizons]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        tss_vals = [metrics[h]["best_tss"] for h in horizons]
        axes[0].bar(h_labels, tss_vals, color="steelblue", alpha=0.8)
        axes[0].axhline(y=0.74, color="red", linestyle="--",
                        label="Hassani et al. 2025 benchmark (TSS=0.74)")
        axes[0].set_xlabel("Forecast Horizon (min)", fontsize=11)
        axes[0].set_ylabel("TSS", fontsize=11)
        axes[0].set_title("True Skill Statistic", fontsize=12, fontweight="bold")
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1)

        auc_vals = [metrics[h]["auc"] for h in horizons]
        axes[1].bar(h_labels, auc_vals, color="forestgreen", alpha=0.8)
        axes[1].axhline(y=0.87, color="red", linestyle="--",
                        label="Hassani et al. 2025 benchmark (AUC=0.87)")
        axes[1].set_xlabel("Forecast Horizon (min)", fontsize=11)
        axes[1].set_ylabel("AUC", fontsize=11)
        axes[1].set_title("AUC-ROC", fontsize=12, fontweight="bold")
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0.5, 1)

        if "mean_lead_time" in metrics[horizons[0]]:
            lt_vals = [metrics[h]["mean_lead_time"] for h in horizons]
            axes[2].bar(h_labels, lt_vals, color="darkorange", alpha=0.8)
            axes[2].set_xlabel("Forecast Horizon (min)", fontsize=11)
            axes[2].set_ylabel("Mean Lead Time (min)", fontsize=11)
            axes[2].set_title("Forecast Lead Time", fontsize=12,
                              fontweight="bold")
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved metrics plot to {save_path}")
        plt.close(fig)
        return fig

    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, horizon: int,
                              save_path: Optional[str] = None):
        """Plot confusion matrix for a given horizon."""
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No Flare", "Flare"])
        ax.set_yticklabels(["No Flare", "Flare"])
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.set_title(f"Confusion Matrix (Horizon: {horizon} min)",
                     fontsize=12, fontweight="bold")

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                        fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")

        plt.colorbar(im)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig
