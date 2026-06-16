"""
Model inference wrapper for Streamlit app.

Loads the trained ParallelFlareModel checkpoint and runs sliding-window
inference on preprocessed feature data.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.models.cnn_bilstm_attention import ParallelFlareModel

logger = logging.getLogger(__name__)

HORIZONS = [15, 30, 60]
HORIZON_LABELS = ["15 min", "30 min", "60 min"]


def load_model(
    checkpoint_path: str,
    config: dict,
    device: torch.device = None,
) -> Optional[ParallelFlareModel]:
    """Load a trained model from checkpoint.

    Returns None if checkpoint doesn't exist or is invalid.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = Path(checkpoint_path)
    if not path.exists():
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return None

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        feat_cfg = config["features"]
        n_solexs = len(feat_cfg.get("soft_channels", []))
        n_hel1os = len(feat_cfg.get("hard_channels", []))

        model = ParallelFlareModel(n_solexs, n_hel1os, config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        logger.info(f"Loaded model from {checkpoint_path} "
                    f"(epoch={checkpoint.get('epoch', '?')}, "
                    f"val_tss={checkpoint.get('val_tss', '?'):.4f})")
        return model

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return None


@torch.no_grad()
def run_inference(
    model: ParallelFlareModel,
    features: np.ndarray,
    timestamps: pd.DatetimeIndex,
    config: dict,
    device: torch.device = None,
) -> pd.DataFrame:
    """Run sliding-window inference on feature data.

    Args:
        model: Trained ParallelFlareModel
        features: [N, F] array of preprocessed features
        timestamps: DatetimeIndex of length N
        config: Full config dict
        device: Torch device

    Returns:
        DataFrame with columns: timestamp, prob_15min, prob_30min, prob_60min
    """
    if device is None:
        device = next(model.parameters()).device

    window_steps = config["data"].get("input_window_steps", 360)
    stride_steps = config["data"].get("sliding_stride_steps", 30)

    N = len(features)
    if N < window_steps:
        logger.warning(f"Not enough data ({N} steps) for window ({window_steps})")
        return pd.DataFrame()

    predictions = []
    pred_times = []

    for start in range(0, N - window_steps, stride_steps):
        end = start + window_steps
        window = features[start:end]

        # Handle NaN/inf
        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)

        x = torch.FloatTensor(window).unsqueeze(0).to(device)  # [1, T, F]
        probs = model(x).cpu().numpy().flatten()  # [3]

        predictions.append(probs)
        pred_times.append(timestamps[end - 1])

    if not predictions:
        return pd.DataFrame()

    preds = np.array(predictions)

    result = pd.DataFrame({
        "timestamp": pred_times,
        "prob_15min": preds[:, 0],
        "prob_30min": preds[:, 1],
        "prob_60min": preds[:, 2],
    })

    logger.info(f"Inference complete: {len(result)} predictions over "
                f"{len(features)} steps")

    return result


def get_forecast_summary(predictions: pd.DataFrame) -> Dict:
    """Summarize forecast probabilities for display.

    Returns dict with max probabilities, alert level, and recent values.
    """
    if predictions.empty:
        return {
            "max_15min": 0.0,
            "max_30min": 0.0,
            "max_60min": 0.0,
            "alert_level": "low",
            "alert_color": "green",
            "recent_15min": 0.0,
            "recent_30min": 0.0,
            "recent_60min": 0.0,
        }

    max_15 = float(predictions["prob_15min"].max())
    max_30 = float(predictions["prob_30min"].max())
    max_60 = float(predictions["prob_60min"].max())

    recent_15 = float(predictions["prob_15min"].iloc[-1])
    recent_30 = float(predictions["prob_30min"].iloc[-1])
    recent_60 = float(predictions["prob_60min"].iloc[-1])

    max_prob = max(max_15, max_30, max_60)

    if max_prob > 0.7:
        alert_level = "high"
        alert_color = "red"
    elif max_prob > 0.3:
        alert_level = "moderate"
        alert_color = "orange"
    else:
        alert_level = "low"
        alert_color = "green"

    return {
        "max_15min": max_15,
        "max_30min": max_30,
        "max_60min": max_60,
        "alert_level": alert_level,
        "alert_color": alert_color,
        "recent_15min": recent_15,
        "recent_30min": recent_30,
        "recent_60min": recent_60,
    }
