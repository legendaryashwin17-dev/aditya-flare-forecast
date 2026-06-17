"""Premium forecast page — live dashboard with Plotly charts and alerts."""

import os
import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import yaml
from pathlib import Path

from src.app.utils.fits_reader import read_pair, read_solexs_lc, read_hel1os_lc
from src.app.utils.inference import load_model, run_inference, get_forecast_summary
from src.app.utils.pradan_download import PRADANDownloader
from src.app.components.charts import (
    plot_light_curves, plot_forecast_bars, plot_hardness_ratio,
)
from src.app.components.alerts import show_forecast_alert, show_flare_catalogue
from src.data.preprocessing import FlarePreprocessor


def load_config():
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {
        "data": {"input_window_steps": 360, "sliding_stride_steps": 30,
                 "forecast_horizons_minutes": [15, 30, 60]},
        "preprocessing": {"binning_cadence": 10, "fill_method": "ffill",
                          "max_gap_seconds": 300, "background_percentile": 10},
        "features": {
            "soft_channels": ["solexs_flux"],
            "hard_channels": ["hel1os_flux"],
        },
        "model": {"architecture": "Parallel1DCNN-BiLSTM-MultiHead",
                  "solexs_branch": {"filters": 32, "kernel_size": 3, "activation": "relu", "padding": "same"},
                  "hel1os_branch": {"filters": 32, "kernel_size": 3, "activation": "relu", "padding": "same"},
                  "overlap_conv": {"filters": 64, "kernel_size": 5, "activation": "relu", "padding": "same"},
                  "lstm": {"layers": 1, "hidden_size": 64, "dropout": 0.0, "bidirectional": True},
                  "heads": {"units": [32], "dropout": 0.4, "activation": "relu", "n_horizons": 3}},
    }


def _check_model():
    """Return (exists, info) for the trained model checkpoint."""
    ckpt = Path(__file__).parent.parent.parent.parent / "data" / "models" / "best_model.pt"
    if not ckpt.exists():
        return False, None
    try:
        import torch
        ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=False)
        return True, {
            "epoch": ckpt_data.get("epoch"),
            "val_tss": ckpt_data.get("val_tss"),
            "config": ckpt_data.get("config"),
        }
    except Exception:
        return True, None


def _load_real_data(settings):
    """Try to load real data from PRADAN download or FITS upload."""
    solexs_df = None
    hel1os_df = None
    data_source = None

    if settings["data_mode"] == "Download from PRADAN":
        pradan_user = settings.get("pradan_user") or os.environ.get("PRADAN_USERNAME", "")
        pradan_pass = settings.get("pradan_pass") or os.environ.get("PRADAN_PASSWORD", "")

        if pradan_user and pradan_pass:
            if st.button("Download Latest Data from PRADAN", type="primary"):
                with st.spinner("Connecting to PRADAN..."):
                    try:
                        downloader = PRADANDownloader(
                            username=pradan_user,
                            password=pradan_pass,
                            download_dir="data/pradan",
                        )
                        if downloader.login():
                            st.success("Logged in to PRADAN")
                            with st.spinner("Downloading..."):
                                solexs_paths, hel1os_paths = downloader.download_latest(1, 1)
                                if solexs_paths or hel1os_paths:
                                    st.success(f"Downloaded: {len(solexs_paths)} SoLEXS, {len(hel1os_paths)} HEL1OS")
                                    data_source = "pradan"
                                    try:
                                        if solexs_paths:
                                            solexs_df = read_solexs_lc(str(solexs_paths[0]))
                                        if hel1os_paths:
                                            hel1os_df = read_hel1os_lc(str(hel1os_paths[0]))
                                    except Exception as e:
                                        st.error(f"Error reading FITS: {e}")
                                else:
                                    st.warning("No files found on PRADAN.")
                        else:
                            st.error("Login failed. Check credentials.")
                    except Exception as e:
                        st.error(f"Download error: {e}")
        else:
            st.info("Enter PRADAN credentials in the sidebar to download real data.")

    elif settings["data_mode"] == "Upload FITS Files":
        if settings["solexs_file"] or settings["hel1os_file"]:
            with st.spinner("Reading FITS files..."):
                solexs_path = hel1os_path = None

                if settings["solexs_file"]:
                    sname = settings["solexs_file"].name.lower()
                    if sname.endswith((".gti", ".evt", ".pha", ".spec")):
                        st.error(f"Rejected '{settings['solexs_file'].name}': "
                                 "This is not a light curve file. Upload a .lc file.")
                    else:
                        with tempfile.NamedTemporaryFile(
                            suffix=".fits", delete=False, prefix="solexs_",
                        ) as f:
                            f.write(settings["solexs_file"].getvalue())
                            solexs_path = f.name

                if settings["hel1os_file"]:
                    hname = settings["hel1os_file"].name.lower()
                    if hname.endswith((".gti", ".evt", ".pha", ".spec")):
                        st.error(f"Rejected '{settings['hel1os_file'].name}': "
                                 "This is not a light curve file.")
                    elif "lightcurve" not in hname:
                        st.warning(f"'{settings['hel1os_file'].name}' doesn't look like "
                                   "a lightcurve file. Will attempt to read anyway...")
                        with tempfile.NamedTemporaryFile(
                            suffix=".fits", delete=False, prefix="hel1os_",
                        ) as f:
                            f.write(settings["hel1os_file"].getvalue())
                            hel1os_path = f.name
                    else:
                        with tempfile.NamedTemporaryFile(
                            suffix=".fits", delete=False, prefix="hel1os_",
                        ) as f:
                            f.write(settings["hel1os_file"].getvalue())
                            hel1os_path = f.name

                if solexs_path or hel1os_path:
                    try:
                        solexs_df, hel1os_df = read_pair(solexs_path, hel1os_path)
                        data_source = "upload"
                        st.success(f"Loaded: SoLEXS {len(solexs_df)} pts, HEL1OS {len(hel1os_df)} pts")
                    except Exception as e:
                        st.error(f"Error reading FITS: {e}")

    # Check for pre-existing PRADAN data in data/ folder
    if solexs_df is None and hel1os_df is None:
        pradan_solexs = Path("data/pradan_solexs")
        pradan_hel1os = Path("data/pradan_hel1os")
        if pradan_solexs.exists() and pradan_hel1os.exists():
            solexs_files = list(pradan_solexs.glob("**/*.lc"))
            hel1os_files = list(pradan_hel1os.glob("**/lightcurve_czt*.fits"))
            if solexs_files and hel1os_files:
                try:
                    solexs_df = read_solexs_lc(str(solexs_files[0]))
                    hel1os_df = read_hel1os_lc(str(hel1os_files[0]))
                    data_source = "cached"
                except Exception:
                    pass

    return solexs_df, hel1os_df, data_source


def _run_forecast(solexs_df, hel1os_df, config, model_info, flare_catalogue=None):
    """Run inference and display results."""
    preprocessor = FlarePreprocessor(config)
    combined = preprocessor.unify_and_bin(solexs_df, hel1os_df, target_cadence_s=10)
    combined = preprocessor.handle_gaps(combined)
    combined = preprocessor.compute_features(combined)
    combined, feature_params = preprocessor.standardize(combined)
    feature_cols = [c for c in combined.columns if c != "is_valid"]
    features = combined[feature_cols].values.astype(np.float32)
    timestamps = combined.index

    checkpoint_path = str(Path(__file__).parent.parent.parent.parent / "data" / "models" / "best_model.pt")
    model = load_model(checkpoint_path, config)

    if model is None:
        st.error("Failed to load model checkpoint.")
        return

    with st.spinner("Running inference..."):
        predictions = run_inference(model, features, timestamps, config)

    if predictions.empty:
        st.warning("Not enough data for inference. Need at least 60 minutes of continuous data.")
        return

    summary = get_forecast_summary(predictions)
    show_forecast_alert(summary)

    fig = plot_light_curves(solexs_df, hel1os_df, predictions=predictions,
                            flare_catalogue=flare_catalogue)
    st.plotly_chart(fig, use_container_width=True)

    fig_hr = plot_hardness_ratio(solexs_df, hel1os_df)
    st.plotly_chart(fig_hr, use_container_width=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        fig_bars = plot_forecast_bars(summary)
        st.plotly_chart(fig_bars, use_container_width=True)
    with col2:
        epoch_str = f"Epoch {model_info['epoch']}" if model_info and model_info.get("epoch") else "—"
        tss_str = f"{model_info['val_tss']:.3f}" if model_info and model_info.get("val_tss") is not None else "—"
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-item">
                <div class="metric-val">CNN-BiLSTM</div>
                <div class="metric-label">Architecture</div>
            </div>
            <div class="metric-item">
                <div class="metric-val">~102K</div>
                <div class="metric-label">Parameters</div>
            </div>
            <div class="metric-item">
                <div class="metric-val">{epoch_str}</div>
                <div class="metric-label">Trained</div>
            </div>
            <div class="metric-item">
                <div class="metric-val">{tss_str}</div>
                <div class="metric-label">Val TSS</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if flare_catalogue is not None and not flare_catalogue.empty:
        show_flare_catalogue(flare_catalogue)

    with st.expander("Raw Predictions (last 20 rows)"):
        st.dataframe(
            predictions.tail(20).style.format({
                "prob_15min": "{:.4f}", "prob_30min": "{:.4f}", "prob_60min": "{:.4f}",
            }),
            use_container_width=True, hide_index=True,
        )


def _show_skeleton():
    """Show skeleton loading state."""
    st.markdown("""
    <div class="skeleton skeleton-title"></div>
    <div class="skeleton skeleton-text" style="width:80%"></div>
    <div class="skeleton skeleton-text" style="width:60%"></div>
    <div class="skeleton skeleton-chart"></div>
    <div class="skeleton skeleton-chart" style="height:200px"></div>
    """, unsafe_allow_html=True)


def _show_empty_state():
    """Show empty state when no data is loaded."""
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon" style="font-size:3rem;margin-bottom:1rem;">
            <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"
                 style="color:var(--accent);">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/>
                <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
        </div>
        <h3 class="empty-title">No Data Loaded</h3>
        <p class="empty-desc">
            Upload FITS files or download from PRADAN to see real flare forecasts.
            Required files: SoLEXS (.lc) and HEL1OS (lightcurve_*.fits).
        </p>
    </div>
    """, unsafe_allow_html=True)


def _show_no_model():
    """Show state when no trained model exists."""
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon" style="font-size:3rem;margin-bottom:1rem;">
            <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"
                 style="color:var(--warning);">
                <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2Z"/>
                <path d="M12 8v4"/>
                <path d="M12 16h.01"/>
            </svg>
        </div>
        <h3 class="empty-title">No Trained Model</h3>
        <p class="empty-desc">
            Train a model first to get real predictions.
        </p>
        <code>python -m src.train_pradan --mode=train_real</code>
    </div>
    """, unsafe_allow_html=True)


def _show_demo():
    """Show demo mode with simulated data."""
    from src.app.utils.simulated import generate_simulated_data

    st.markdown("""
    <div class="demo-banner">
        <div class="demo-title">Demo Mode</div>
        <div class="demo-desc">
            Showing simulated flare data. Upload real FITS files or download from PRADAN for actual forecasts.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Generating simulated data..."):
        solexs_df, hel1os_df, flare_catalogue = generate_simulated_data(
            duration_hours=24.0, n_flares=5)

    model_exists, model_info = _check_model()

    if not model_exists:
        _show_no_model()
        return

    # Show simulated light curves with model predictions
    fig = plot_light_curves(solexs_df, hel1os_df, flare_catalogue=flare_catalogue)
    st.plotly_chart(fig, use_container_width=True)

    fig_hr = plot_hardness_ratio(solexs_df, hel1os_df)
    st.plotly_chart(fig_hr, use_container_width=True)

    # Show flare catalogue
    if flare_catalogue is not None and not flare_catalogue.empty:
        show_flare_catalogue(flare_catalogue)


def render_forecast_page(settings: dict):
    config = load_config()

    st.markdown('<div class="noise-overlay"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:2rem 0 1rem 0;">
        <div class="section-eyebrow">Live Dashboard</div>
        <h2 class="section-title" style="font-size:clamp(1.5rem,3vw,2.2rem);">Solar Flare Forecast</h2>
    </div>
    """, unsafe_allow_html=True)

    # Try to load real data
    solexs_df, hel1os_df, data_source = _load_real_data(settings)

    # If no real data, show demo
    if solexs_df is None or hel1os_df is None:
        use_simulated = settings.get("use_simulated", False)
        if use_simulated or data_source is None:
            _show_demo()
            return
        _show_empty_state()
        return

    # Real data loaded — run inference
    model_exists, model_info = _check_model()

    if not model_exists:
        _show_no_model()
        return

    # Show data source indicator
    source_labels = {
        "pradan": ("PRADAN Download", "#00ffaa"),
        "upload": ("FITS Upload", "#00aaff"),
        "cached": ("Cached PRADAN Data", "#ffa502"),
    }
    label, color = source_labels.get(data_source, ("Unknown", "#888"))
    st.markdown(f"""
    <div style="display:inline-flex;align-items:center;gap:0.5rem;padding:0.4rem 0.8rem;
        border-radius:8px;background:rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1);
        border:1px solid {color}33;margin-bottom:1rem;">
        <div style="width:6px;height:6px;border-radius:50%;background:{color};"></div>
        <span style="font-family:'Geist Mono',monospace;font-size:0.7rem;color:{color};">{label}</span>
        <span style="font-size:0.7rem;color:#888;">| SoLEXS: {len(solexs_df)} pts, HEL1OS: {len(hel1os_df)} pts</span>
    </div>
    """, unsafe_allow_html=True)

    _run_forecast(solexs_df, hel1os_df, config, model_info)
