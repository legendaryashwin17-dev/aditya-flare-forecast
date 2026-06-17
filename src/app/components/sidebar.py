"""Premium sidebar component."""

import os
from pathlib import Path
import streamlit as st
import yaml


def _model_status():
    """Check if a trained model checkpoint exists."""
    ckpt = Path(__file__).parent.parent.parent.parent / "data" / "models" / "best_model.pt"
    if not ckpt.exists():
        return False, "No trained model found"
    try:
        import torch
        ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=False)
        epoch = ckpt_data.get("epoch", "?")
        val_tss = ckpt_data.get("val_tss", None)
        tss_str = f", val_TSS={val_tss:.3f}" if val_tss is not None else ""
        return True, f"Epoch {epoch}{tss_str}"
    except Exception:
        return True, "Checkpoint exists (unverified)"


def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding: 1.5rem 0 1rem 0;">
            <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.5rem;">
                <div style="width:36px;height:36px;border-radius:12px;background:linear-gradient(135deg,#00ffaa,#0088ff);
                    display:flex;align-items:center;justify-content:center;font-size:1.1rem;">☀</div>
                <div>
                    <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1rem;
                        color:#e8e8ec;letter-spacing:-0.02em;">Aditya-L1</div>
                    <div style="font-family:'Geist Mono',monospace;font-size:0.6rem;letter-spacing:0.1em;
                        text-transform:uppercase;color:#4a4a5a;">Flare Forecast</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="height:1px;background:rgba(255,255,255,0.06);margin:0.5rem 0;"></div>',
                    unsafe_allow_html=True)

        # --- Model status ---
        model_ok, model_msg = _model_status()
        if model_ok:
            st.markdown(f"""
            <div style="padding:0.5rem 0.75rem;border-radius:8px;background:rgba(0,255,170,0.08);
                border:1px solid rgba(0,255,170,0.2);margin-bottom:0.75rem;">
                <div style="font-family:'Geist Mono',monospace;font-size:0.6rem;letter-spacing:0.1em;
                    text-transform:uppercase;color:#00ffaa;">Model Ready</div>
                <div style="font-size:0.75rem;color:#888;margin-top:0.25rem;">{model_msg}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="padding:0.5rem 0.75rem;border-radius:8px;background:rgba(255,100,100,0.08);
                border:1px solid rgba(255,100,100,0.2);margin-bottom:0.75rem;">
                <div style="font-family:'Geist Mono',monospace;font-size:0.6rem;letter-spacing:0.1em;
                    text-transform:uppercase;color:#ff6464;">No Model</div>
                <div style="font-size:0.75rem;color:#888;margin-top:0.25rem;">Train or upload a checkpoint</div>
            </div>
            """, unsafe_allow_html=True)

        # --- Data Source ---
        data_mode = st.radio(
            "Data Source",
            ["Upload FITS Files", "Download from PRADAN"],
            help="Provide real Aditya-L1 data for forecasting",
        )

        solexs_file = None
        hel1os_file = None
        pradan_user = None
        pradan_pass = None

        if data_mode == "Download from PRADAN":
            st.markdown('<div style="font-family:\'Geist Mono\',monospace;font-size:0.65rem;letter-spacing:0.1em;'
                        'text-transform:uppercase;color:#4a4a5a;margin:0.75rem 0 0.5rem 0;">PRADAN Credentials</div>',
                        unsafe_allow_html=True)
            pradan_user = st.text_input(
                "Username",
                value=os.environ.get("PRADAN_USERNAME", ""),
                placeholder="Enter your PRADAN username",
            )
            pradan_pass = st.text_input(
                "Password",
                type="password",
                value=os.environ.get("PRADAN_PASSWORD", ""),
                placeholder="Enter your PRADAN password",
            )

        elif data_mode == "Upload FITS Files":
            st.markdown('<div style="font-family:\'Geist Mono\',monospace;font-size:0.65rem;letter-spacing:0.1em;'
                        'text-transform:uppercase;color:#4a4a5a;margin:0.75rem 0 0.5rem 0;">Upload</div>',
                        unsafe_allow_html=True)
            solexs_file = st.file_uploader(
                "SoLEXS FITS (.lc light curve only)",
                type=["fits", "fit"],
                help="Upload AL1_SOLEXS_*_L1.lc — NOT .gti or .evt files",
            )
            hel1os_file = st.file_uploader(
                "HEL1OS FITS (lightcurve_*.fits only)",
                type=["fits", "fit"],
                help="Upload lightcurve_cdte*.fits or lightcurve_czt*.fits",
            )

        st.markdown('<div style="height:1px;background:rgba(255,255,255,0.06);margin:0.75rem 0;"></div>',
                    unsafe_allow_html=True)

        with st.expander("Advanced Settings"):
            confidence_threshold = st.slider("Alert Threshold", 0.1, 0.9, 0.3, 0.05)
            window_minutes = st.slider("Input Window (min)", 30, 120, 60, 15)

        st.markdown('<div style="height:1px;background:rgba(255,255,255,0.06);margin:0.75rem 0;"></div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div style="font-family:'Geist Mono',monospace;font-size:0.6rem;letter-spacing:0.1em;
            text-transform:uppercase;color:#4a4a5a;text-align:center;padding:0.5rem 0;">
            Bharatiya Antariksh Hackathon 2026
        </div>
        """, unsafe_allow_html=True)

    return {
        "data_mode": data_mode,
        "solexs_file": solexs_file,
        "hel1os_file": hel1os_file,
        "pradan_user": pradan_user,
        "pradan_pass": pradan_pass,
        "model_mode": "Pre-trained Checkpoint",
        "confidence_threshold": confidence_threshold,
        "window_minutes": window_minutes,
        "use_simulated": True,
    }
