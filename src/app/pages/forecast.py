"""
Main forecast page — the core demo of the Aditya-L1 Solar Flare Forecast.
"""

import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import yaml
from pathlib import Path

from src.app.utils.fits_reader import read_pair, read_solexs_lc, read_hel1os_lc, extract_zip, find_fits_in_dir
from src.app.utils.simulated import generate_simulated_data
from src.app.utils.inference import load_model, run_inference, get_forecast_summary
from src.app.utils.pradan_download import PRADANDownloader
from src.app.components.charts import (
    plot_light_curves,
    plot_forecast_bars,
    plot_hardness_ratio,
)
from src.app.components.alerts import (
    show_forecast_alert,
    show_flare_catalogue,
    show_model_info,
)
from src.data.preprocessing import FlarePreprocessor


def load_config():
    """Load the project config."""
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    # Default config if file not found
    return {
        "data": {
            "input_window_steps": 360,
            "sliding_stride_steps": 30,
            "forecast_horizons_minutes": [15, 30, 60],
        },
        "preprocessing": {
            "binning_cadence": 10,
            "fill_method": "ffill",
            "max_gap_seconds": 300,
            "background_percentile": 10,
        },
        "features": {
            "soft_channels": [
                "solexs_flux", "solexs_flux_log", "dsoft_dt_30s", "dsoft_dt_2min",
                "soft_rolling_mean_1min", "soft_rolling_std_1min", "background_subtracted_soft"
            ],
            "hard_channels": [
                "hel1os_flux", "hel1os_flux_log", "spectral_hardness_ratio",
                "dhard_dt_30s", "dhard_dt_2min", "overlap_xcorr",
                "hard_rolling_mean_1min", "hard_rolling_std_1min",
                "hard_rolling_mean_5min", "background_subtracted_hard"
            ],
        },
        "model": {
            "architecture": "Parallel1DCNN-BiLSTM-MultiHead",
            "solexs_branch": {"filters": 32, "kernel_size": 3, "activation": "relu", "padding": "same"},
            "hel1os_branch": {"filters": 32, "kernel_size": 3, "activation": "relu", "padding": "same"},
            "overlap_conv": {"filters": 64, "kernel_size": 5, "activation": "relu", "padding": "same"},
            "lstm": {"layers": 1, "hidden_size": 64, "dropout": 0.0, "bidirectional": True},
            "heads": {"units": [32], "dropout": 0.4, "activation": "relu", "n_horizons": 3},
        },
    }


def render_forecast_page(settings: dict):
    """Render the main forecast page."""
    config = load_config()

    st.header("Solar Flare Forecast")
    st.caption("Upload data or use simulated flares to see real-time predictions")

    # --- Step 1: Load Data ---
    solexs_df = None
    hel1os_df = None
    flare_catalogue = None

    if settings["data_mode"] == "Upload FITS Files":
        if settings["solexs_file"] or settings["hel1os_file"]:
            with st.spinner("Reading FITS files..."):
                solexs_path = None
                hel1os_path = None

                if settings["solexs_file"]:
                    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
                        f.write(settings["solexs_file"].getvalue())
                        solexs_path = f.name

                if settings["hel1os_file"]:
                    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
                        f.write(settings["hel1os_file"].getvalue())
                        hel1os_path = f.name

                try:
                    solexs_df, hel1os_df = read_pair(solexs_path, hel1os_path)
                    st.success(f"Loaded: SoLEXS {len(solexs_df)} pts, HEL1OS {len(hel1os_df)} pts")
                except Exception as e:
                    st.error(f"Error reading FITS: {e}")
                    st.info("Falling back to simulated data...")
                    settings["data_mode"] = "Simulated Flares"
        else:
            st.info("Upload FITS files to analyze real Aditya-L1 data, or use simulated data below.")
            settings["data_mode"] = "Simulated Flares"

    if settings["data_mode"] == "Download from PRADAN":
        if settings["pradan_user"] and settings["pradan_pass"]:
            if st.button("Download Latest Data from PRADAN", type="primary"):
                with st.spinner("Connecting to PRADAN portal..."):
                    try:
                        downloader = PRADANDownloader(
                            username=settings["pradan_user"],
                            password=settings["pradan_pass"],
                            download_dir="data/pradan",
                        )
                        if downloader.login():
                            st.success("Logged in to PRADAN successfully!")
                            with st.spinner("Downloading latest SoLEXS + HEL1OS data..."):
                                solexs_paths, hel1os_paths = downloader.download_latest(
                                    n_solexs=1, n_hel1os=1
                                )
                                if solexs_paths or hel1os_paths:
                                    st.success(
                                        f"Downloaded: {len(solexs_paths)} SoLEXS, "
                                        f"{len(hel1os_paths)} HEL1OS files"
                                    )
                                    # Read the downloaded files
                                    try:
                                        if solexs_paths:
                                            solexs_df = read_solexs_lc(str(solexs_paths[0]))
                                        if hel1os_paths:
                                            hel1os_df = read_hel1os_lc(str(hel1os_paths[0]))
                                    except Exception as e:
                                        st.error(f"Error reading downloaded FITS: {e}")
                                else:
                                    st.warning("No files found on PRADAN for download.")
                        else:
                            st.error("Failed to log in to PRADAN. Check credentials.")
                    except Exception as e:
                        st.error(f"PRADAN download error: {e}")
                        st.info("Falling back to simulated data...")
                        settings["data_mode"] = "Simulated Flares"
        else:
            st.info("Enter PRADAN credentials in the sidebar to download data.")
            settings["data_mode"] = "Simulated Flares"

    if settings["data_mode"] == "Simulated Flares":
        with st.spinner("Generating simulated flare data..."):
            solexs_df, hel1os_df, flare_catalogue = generate_simulated_data(
                duration_hours=24.0,
                n_flares=5,
                seed=42,
            )
            st.info(f"Generated 24h of simulated data with {len(flare_catalogue)} flare events")

    if solexs_df is None or hel1os_df is None:
        st.warning("No data available. Please upload FITS files or use simulated data.")
        return

    # --- Step 2: Preprocess ---
    with st.spinner("Preprocessing data (binning, features, standardization)..."):
        preprocessor = FlarePreprocessor(config)

        # Unify to common time grid
        combined = preprocessor.unify_and_bin(solexs_df, hel1os_df, target_cadence_s=10)
        combined = preprocessor.handle_gaps(combined)
        combined = preprocessor.compute_features(combined)
        combined, feature_params = preprocessor.standardize(combined)

        feature_cols = [c for c in combined.columns if c != "is_valid"]
        features = combined[feature_cols].values.astype(np.float32)
        timestamps = combined.index

        st.caption(f"Preprocessed: {len(features)} timesteps, {features.shape[1]} features")

    # --- Step 3: Run Inference ---
    checkpoint_path = str(Path(__file__).parent.parent.parent.parent / "data" / "models" / "best_model.pt")
    model = load_model(checkpoint_path, config)

    if model is None:
        st.warning("No trained model found. Showing data visualization only.")
        st.info("Train a model first: `python scripts/train_demo.py`")

        # Show light curves without forecasts
        fig = plot_light_curves(solexs_df, hel1os_df, flare_catalogue=flare_catalogue)
        st.plotly_chart(fig, use_container_width=True)

        if flare_catalogue is not None and not flare_catalogue.empty:
            show_flare_catalogue(flare_catalogue)
        return

    with st.spinner("Running model inference..."):
        predictions = run_inference(model, features, timestamps, config)

    if predictions.empty:
        st.warning("Insufficient data for inference. Need at least 60 minutes of data.")
        return

    # --- Step 4: Display Results ---
    summary = get_forecast_summary(predictions)

    # Alert banner
    show_forecast_alert(summary)

    # Light curves + forecast
    fig = plot_light_curves(
        solexs_df, hel1os_df,
        predictions=predictions,
        flare_catalogue=flare_catalogue,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Hardness ratio
    fig_hr = plot_hardness_ratio(solexs_df, hel1os_df)
    st.plotly_chart(fig_hr, use_container_width=True)

    # Forecast bars
    col1, col2 = st.columns([1, 1])
    with col1:
        fig_bars = plot_forecast_bars(summary)
        st.plotly_chart(fig_bars, use_container_width=True)

    with col2:
        show_model_info({})

    # Flare catalogue
    if flare_catalogue is not None and not flare_catalogue.empty:
        show_flare_catalogue(flare_catalogue)

    # Prediction data
    with st.expander("Raw Predictions (last 20 rows)"):
        st.dataframe(
            predictions.tail(20).style.format({
                "prob_15min": "{:.4f}",
                "prob_30min": "{:.4f}",
                "prob_60min": "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
