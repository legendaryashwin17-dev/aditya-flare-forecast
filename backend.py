"""
FastAPI backend for Aditya-L1 Solar Flare Forecast.

Endpoints:
    POST /api/forecast       — Upload FITS files, run inference, return predictions
    POST /api/forecast/simulated — Generate simulated forecast for demo
    GET  /api/status         — Health check
    GET  /api/model-info     — Model metadata and training results
"""

import io
import json
import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.app.utils.fits_reader import read_pair, detect_instrument
from src.app.utils.inference import load_model, run_inference, get_forecast_summary
from src.app.utils.simulated import generate_simulated_data
from src.data.preprocessing import FlarePreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Aditya-L1 Solar Flare Forecast API",
    description="AI-powered solar flare nowcasting and forecasting",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_config():
    config_path = Path(__file__).parent / "config" / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {
        "data": {"input_window_steps": 360, "sliding_stride_steps": 30,
                 "forecast_horizons_minutes": [15, 30, 60]},
        "preprocessing": {"binning_cadence": 10, "fill_method": "ffill",
                          "max_gap_seconds": 300, "background_percentile": 10},
        "features": {
            "soft_channels": ["solexs_flux", "solexs_flux_log", "dsoft_dt_30s", "dsoft_dt_2min",
                              "soft_rolling_mean_1min", "soft_rolling_std_1min", "background_subtracted_soft"],
            "hard_channels": ["hel1os_flux", "hel1os_flux_log", "spectral_hardness_ratio",
                              "dhard_dt_30s", "dhard_dt_2min", "overlap_xcorr",
                              "hard_rolling_mean_1min", "hard_rolling_std_1min",
                              "hard_rolling_mean_5min", "background_subtracted_hard"],
        },
        "model": {"architecture": "Parallel1DCNN-BiLSTM-MultiHead",
                  "solexs_branch": {"filters": 32, "kernel_size": 3, "activation": "relu", "padding": "same"},
                  "hel1os_branch": {"filters": 32, "kernel_size": 3, "activation": "relu", "padding": "same"},
                  "overlap_conv": {"filters": 64, "kernel_size": 5, "activation": "relu", "padding": "same"},
                  "lstm": {"layers": 1, "hidden_size": 64, "dropout": 0.0, "bidirectional": True},
                  "heads": {"units": [32], "dropout": 0.4, "activation": "relu", "n_horizons": 3}},
    }


@app.get("/api/status")
async def status():
    config = load_config()
    checkpoint = Path(__file__).parent / "data" / "models" / "best_model.pt"
    return {
        "status": "ok",
        "model_available": checkpoint.exists(),
        "model_path": str(checkpoint),
        "architecture": config["model"]["architecture"],
        "horizons": config["data"]["forecast_horizons_minutes"],
    }


@app.get("/api/model-info")
async def model_info():
    results_path = Path(__file__).parent / "data" / "training_results.json"
    results = {}
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

    return {
        "architecture": "Parallel1DCNN-BiLSTM-MultiHead",
        "total_params": 101539,
        "solexs_features": 7,
        "hel1os_features": 10,
        "input_shape": [360, 17],
        "output_shape": [3],
        "horizons_minutes": [15, 30, 60],
        "training": results.get("training", {}),
        "test_metrics": results.get("test_metrics", {}),
    }


@app.post("/api/forecast")
async def forecast(
    solexs_file: Optional[UploadFile] = File(None),
    hel1os_file: Optional[UploadFile] = File(None),
):
    """Upload FITS files and run inference."""
    if not solexs_file and not hel1os_file:
        raise HTTPException(status_code=400, detail="Upload at least one FITS file")

    config = load_config()

    solexs_path = hel1os_path = None

    try:
        if solexs_file:
            content = await solexs_file.read()
            suffix = ".lc" if solexs_file.filename and solexs_file.filename.endswith(".lc") else ".fits"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, prefix="solexs_") as f:
                f.write(content)
                solexs_path = f.name

        if hel1os_file:
            content = await hel1os_file.read()
            with tempfile.NamedTemporaryFile(suffix=".fits", delete=False, prefix="hel1os_") as f:
                f.write(content)
                hel1os_path = f.name

        solexs_df, hel1os_df = read_pair(solexs_path, hel1os_path)

        preprocessor = FlarePreprocessor(config)
        combined = preprocessor.unify_and_bin(solexs_df, hel1os_df, target_cadence_s=10)
        combined = preprocessor.handle_gaps(combined)
        combined = preprocessor.compute_features(combined)
        combined, feature_params = preprocessor.standardize(combined)
        feature_cols = [c for c in combined.columns if c != "is_valid"]
        features = combined[feature_cols].values.astype(np.float32)
        timestamps = combined.index

        checkpoint_path = str(Path(__file__).parent / "data" / "models" / "best_model.pt")
        model = load_model(checkpoint_path, config)

        if model is not None:
            predictions = run_inference(model, features, timestamps, config)
        else:
            predictions = _generate_simulated_predictions(timestamps, None)

        summary = get_forecast_summary(predictions)

        return {
            "status": "ok",
            "source": "uploaded_fits",
            "solexs_points": len(solexs_df) if solexs_df is not None else 0,
            "hel1os_points": len(hel1os_df) if hel1os_df is not None else 0,
            "model_used": model is not None,
            "summary": summary,
            "predictions": predictions.to_dict(orient="records"),
        }

    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if solexs_path:
            Path(solexs_path).unlink(missing_ok=True)
        if hel1os_path:
            Path(hel1os_path).unlink(missing_ok=True)


@app.post("/api/forecast/simulated")
async def forecast_simulated(
    duration_hours: float = 24.0,
    n_flares: int = 5,
    seed: int = 42,
):
    """Generate simulated forecast for demo."""
    config = load_config()

    solexs_df, hel1os_df, flare_catalogue = generate_simulated_data(
        duration_hours=duration_hours, n_flares=n_flares, seed=seed)

    preprocessor = FlarePreprocessor(config)
    combined = preprocessor.unify_and_bin(solexs_df, hel1os_df, target_cadence_s=10)
    combined = preprocessor.handle_gaps(combined)
    combined = preprocessor.compute_features(combined)
    combined, feature_params = preprocessor.standardize(combined)
    feature_cols = [c for c in combined.columns if c != "is_valid"]
    features = combined[feature_cols].values.astype(np.float32)
    timestamps = combined.index

    checkpoint_path = str(Path(__file__).parent / "data" / "models" / "best_model.pt")
    model = load_model(checkpoint_path, config)

    if model is not None:
        predictions = run_inference(model, features, timestamps, config)
    else:
        predictions = _generate_simulated_predictions(timestamps, flare_catalogue)

    summary = get_forecast_summary(predictions)

    light_curve = {
        "timestamps": [str(t) for t in solexs_df.index],
        "solexs_flux": solexs_df["solexs_flux"].tolist(),
        "hel1os_flux": hel1os_df["hel1os_flux"].tolist(),
    }

    catalogue = []
    if flare_catalogue is not None:
        for _, row in flare_catalogue.iterrows():
            catalogue.append({
                "peak_time": str(row["peak_time"]),
                "class": row.get("class", "C"),
                "flux": float(row.get("flux", 1e-5)),
            })

    return {
        "status": "ok",
        "source": "simulated",
        "solexs_points": len(solexs_df),
        "hel1os_points": len(hel1os_df),
        "model_used": model is not None,
        "summary": summary,
        "predictions": predictions.to_dict(orient="records"),
        "light_curve": light_curve,
        "flare_catalogue": catalogue,
    }


def _generate_simulated_predictions(timestamps, flare_catalogue):
    rng = np.random.RandomState(42)
    n_preds = max(1, len(timestamps) // 30)
    idx = np.linspace(0, len(timestamps) - 1, n_preds, dtype=int)
    pred_times = timestamps[idx]

    base_15 = rng.uniform(0.05, 0.25, n_preds)
    base_30 = rng.uniform(0.03, 0.20, n_preds)
    base_60 = rng.uniform(0.02, 0.15, n_preds)

    if flare_catalogue is not None and not flare_catalogue.empty:
        for _, flare in flare_catalogue.iterrows():
            peak = flare["peak_time"]
            dists = np.abs((pred_times - peak).total_seconds())
            mask_15 = dists < 900
            mask_30 = dists < 1800
            mask_60 = dists < 3600
            base_15[mask_15] += rng.uniform(0.4, 0.6, mask_15.sum())
            base_30[mask_30] += rng.uniform(0.3, 0.5, mask_30.sum())
            base_60[mask_60] += rng.uniform(0.2, 0.4, mask_60.sum())

    return pd.DataFrame({
        "timestamp": pred_times,
        "prob_15min": np.clip(base_15, 0, 1),
        "prob_30min": np.clip(base_30, 0, 1),
        "prob_60min": np.clip(base_60, 0, 1),
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
