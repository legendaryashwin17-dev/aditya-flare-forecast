"""
About page — model information, metrics, and references.
"""

import streamlit as st
from pathlib import Path
import json


def render_about():
    """Render the about page."""
    st.header("About This Project")

    st.markdown("""
    ---

    #### Problem Statement

    Build an automated algorithmic pipeline that uses combined **SoLEXS** (soft X-ray, 2-22 keV)
    and **HEL1OS** (hard X-ray, 8-150 keV) time-series data from **Aditya-L1** to:

    - **Nowcast** (real-time detect) solar flares using threshold-based detection
    - **Forecast** (predict) flares with quantifiable lead-time

    ---

    #### Model Architecture

    ```
    Input: [B, 360, 17]  (60-min window at 10s cadence, 17 features)
         |
    +----+----+
    |         |
    SoLEXS   HEL1OS        <- Parallel branches (sensor-specific noise)
    |         |
    1D-Conv   1D-Conv
    (32,k=3)  (32,k=3)
    |         |
    +----+----+
         |
    Overlap Conv          <- 64 filters, k=5 (8-22 keV cross-correlation)
         |
    BiLSTM (64)           <- Single layer, bidirectional
         |
    +----+----+
    |    |    |
    Head1 Head2 Head3      <- Multi-head: 15, 30, 60 min
    Dense(32) -> Drop(0.4) -> Dense(1, sigmoid)
    ```

    **Parameters**: ~102,000 (lightweight, fast inference)

    ---

    #### Features (17 total)

    | Branch | Features |
    |--------|----------|
    | SoLEXS (7) | Flux, log(flux), dF/dt (30s), dF/dt (2min), rolling mean (1min), rolling std (1min), background-subtracted |
    | HEL1OS (10) | Flux, log(flux), hardness ratio, dF/dt (30s), dF/dt (2min), overlap cross-correlation, rolling mean (1min/5min), rolling std (1min), background-subtracted |

    ---

    #### Training Details

    - **Loss**: Focal Loss (α=0.75, γ=2.0) for class imbalance
    - **Optimizer**: Adam (lr=1e-3, weight_decay=1e-4)
    - **LR Schedule**: ReduceLROnPlateau (factor=0.5, patience=3)
    - **Early Stopping**: Monitor val TSS, patience=10
    - **Transfer Learning**: Pre-train on GOES XRS (2003-2023), fine-tune on Aditya-L1

    ---

    #### References

    1. **Nandi et al. 2025** — HEL1OS pre-flare precursor brightening in 8-22 keV band
    2. **Sarwade et al. 2025** — SoLEXS instrument design and calibration
    3. **Hassani et al. 2025** — TSS=0.74 benchmark on GOES XRS (2003-2023)
    4. **Lin et al. 2017** — Focal Loss for Dense Object Detection
    5. **Neupert Effect** — d(soft X-ray)/dt tracks hard X-ray flux

    ---

    #### Data Source

    **Aditya-L1** Science Data from **ISSDC PRADAN** portal:
    [https://pradan.issdc.gov.in/al1](https://pradan.issdc.gov.in/al1)

    """)

    # Show training results if available
    results_path = Path(__file__).parent.parent.parent.parent / "data" / "training_results.json"
    if results_path.exists():
        st.subheader("Training Results")
        with open(results_path) as f:
            results = json.load(f)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean TSS", f"{results.get('test_mean_tss', 0):.4f}")
        with col2:
            st.metric("Mean AUC", f"{results.get('test_mean_auc', 0):.4f}")
        with col3:
            st.metric("Best Val TSS", f"{results.get('best_val_tss', 0):.4f}")

        if "horizons" in results:
            st.json(results["horizons"])

    st.markdown("---")
    st.caption("Bharatiya Antariksh Hackathon 2026 — Aditya-L1 Solar Flare Forecast")
