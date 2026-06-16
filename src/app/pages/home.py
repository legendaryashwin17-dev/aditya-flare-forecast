"""
Home page — landing page for the Aditya-L1 Solar Flare Forecast app.
"""

import streamlit as st


def render_home():
    """Render the home/landing page."""
    st.markdown("""
    # Aditya-L1 Solar Flare Forecast

    ### Real-Time Flare Detection and Prediction from SoLEXS + HEL1OS

    ---

    This tool provides **real-time nowcasting** and **short-term forecasting** of solar flares
    using X-ray data from the **Aditya-L1** spacecraft's **SoLEXS** (2-22 keV) and
    **HEL1OS** (8-150 keV) instruments.

    ---

    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lead Time", "15/30/60 min", help="Multi-horizon forecast")
    with col2:
        st.metric("Instruments", "2", help="SoLEXS + HEL1OS")
    with col3:
        st.metric("Architecture", "CNN-BiLSTM", help="Parallel branches with attention")

    st.markdown("---")

    st.markdown("""
    #### How It Works

    1. **Data Ingestion** — SoLEXS and HEL1OS Level-1 FITS files from ISSDC PRADAN
    2. **Preprocessing** — 10s binning, 17 physics-informed features (hardness ratio, dF/dt, cross-correlation)
    3. **Model Inference** — Parallel 1D-CNN branches + BiLSTM + multi-head output
    4. **Forecast** — Probability of flare occurrence at 15, 30, and 60 minute horizons

    ---

    #### Key Innovation: 8-22 keV Overlap Band

    The model focuses on the **8-22 keV overlap** between SoLEXS and HEL1OS,
    where **pre-flare precursor brightening** has been observed (Nandi et al. 2025).
    This band provides predictive information 15-25 minutes before flare onset.

    ---

    #### Getting Started

    - **Simulated Data**: Click "Forecast" in the sidebar and select "Simulated Flares"
    - **Real Data**: Download FITS files from [ISSDC PRADAN](https://pradan.issdc.gov.in/al1)
      and upload them in the Forecast page

    ---
    """)

    st.info(
        "**Bharatiya Antariksh Hackathon 2026** — Problem Statement 15: "
        "Automated algorithmic pipeline for solar flare forecasting from Aditya-L1 data."
    )
