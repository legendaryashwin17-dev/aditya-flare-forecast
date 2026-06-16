"""Premium alert banner components."""

import streamlit as st


def show_forecast_alert(summary: dict):
    alert_level = summary.get("alert_level", "low")
    max_prob = max(
        summary.get("max_15min", 0),
        summary.get("max_30min", 0),
        summary.get("max_60min", 0),
    )

    if alert_level == "high":
        icon = "!"
        cls = "alert-high"
        label = "HIGH FLARE RISK"
    elif alert_level == "moderate":
        icon = "~"
        cls = "alert-moderate"
        label = "MODERATE FLARE RISK"
    else:
        icon = ""
        cls = "alert-low"
        label = "LOW FLARE RISK"

    st.markdown(f"""
    <div class="alert-banner {cls}">
        <div class="alert-icon">{icon}</div>
        <div class="alert-text">
            <div class="alert-label">{label} — Max: {max_prob:.1%}</div>
            <div class="alert-detail">
                15min: {summary['recent_15min']:.1%} |
                30min: {summary['recent_30min']:.1%} |
                60min: {summary['recent_60min']:.1%}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def show_flare_catalogue(cat_df):
    if cat_df is None or cat_df.empty:
        return

    st.markdown("""
    <div class="section-header" style="margin-top:2rem;">
        <div class="section-eyebrow">Events</div>
        <h3 class="section-title" style="font-size:1.1rem;">Detected Flare Events</h3>
    </div>
    """, unsafe_allow_html=True)

    display_df = cat_df[["start_time", "peak_time", "end_time", "goes_class", "peak_flux"]].copy()
    display_df["peak_flux"] = display_df["peak_flux"].apply(lambda x: f"{x:.2e}")
    display_df.columns = ["Start", "Peak", "End", "Class", "Peak Flux (W/m2)"]

    st.dataframe(display_df, use_container_width=True, hide_index=True)


def show_model_info(checkpoint_info: dict):
    st.markdown("""
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
            <div class="metric-val">3</div>
            <div class="metric-label">Heads</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
