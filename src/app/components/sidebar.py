"""
Sidebar component for file upload, PRADAN download, and controls.
"""

import streamlit as st
from pathlib import Path


def render_sidebar():
    """Render the sidebar with file upload, PRADAN download, and controls.

    Returns dict with user selections.
    """
    with st.sidebar:
        st.image(
            "https://www.ursc.gov.in/sites/default/files/uploads/hel1os_0.png",
            width=200,
            use_container_width=True,
        )
        st.title("Aditya-L1 Flare Forecast")
        st.caption("SoLEXS + HEL1OS Real-Time Analysis")

        st.divider()

        # Data source selection
        data_mode = st.radio(
            "Data Source",
            ["Simulated Flares", "Download from PRADAN", "Upload FITS Files"],
            help="Use simulated data, download from PRADAN, or upload your own FITS files",
        )

        solexs_file = None
        hel1os_file = None
        pradan_user = None
        pradan_pass = None

        if data_mode == "Download from PRADAN":
            st.subheader("PRADAN Credentials")
            pradan_user = st.text_input(
                "Username",
                value="ashwani___chaurasia",
                help="ISSDC PRADAN portal username",
            )
            pradan_pass = st.text_input(
                "Password",
                type="password",
                value="Carnage.acsb007",
                help="ISSDC PRADAN portal password",
            )
            st.info(
                "Downloads latest SoLEXS + HEL1OS data from "
                "[PRADAN](https://pradan.issdc.gov.in). "
                "FITS files are saved to `data/pradan/`."
            )

        elif data_mode == "Upload FITS Files":
            st.subheader("Upload Data Files")
            solexs_file = st.file_uploader(
                "SoLEXS FITS (2-22 keV)",
                type=["fits", "fit"],
                help="Level-1 SoLEXS data from ISSDC PRADAN",
            )
            hel1os_file = st.file_uploader(
                "HEL1OS FITS (8-150 keV)",
                type=["fits", "fit"],
                help="Level-1 HEL1OS data from ISSDC PRADAN",
            )

        st.divider()

        # Model selection
        model_mode = st.radio(
            "Model",
            ["Pre-trained Checkpoint", "Retrain on Data"],
            help="Use saved model or train on uploaded data",
        )

        st.divider()

        # Advanced options
        with st.expander("Advanced Options"):
            confidence_threshold = st.slider(
                "Alert Threshold",
                min_value=0.1,
                max_value=0.9,
                value=0.3,
                step=0.05,
                help="Probability above which to show alert",
            )
            window_minutes = st.slider(
                "Input Window (min)",
                min_value=30,
                max_value=120,
                value=60,
                step=15,
                help="Lookback window for model input",
            )

        st.divider()
        st.caption("Bharatiya Antariksh Hackathon 2026")

    return {
        "data_mode": data_mode,
        "solexs_file": solexs_file,
        "hel1os_file": hel1os_file,
        "pradan_user": pradan_user,
        "pradan_pass": pradan_pass,
        "model_mode": model_mode,
        "confidence_threshold": confidence_threshold,
        "window_minutes": window_minutes,
    }
