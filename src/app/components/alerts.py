"""
Alert banner components for the Streamlit app.
"""

import streamlit as st


def show_forecast_alert(summary: dict):
    """Display a color-coded alert banner based on forecast probabilities."""
    alert_level = summary.get("alert_level", "low")
    max_prob = max(
        summary.get("max_15min", 0),
        summary.get("max_30min", 0),
        summary.get("max_60min", 0),
    )

    if alert_level == "high":
        st.error(
            f"**HIGH FLARE RISK** — Maximum probability: **{max_prob:.1%}**\n\n"
            f"Current 15-min: {summary['recent_15min']:.1%} | "
            f"30-min: {summary['recent_30min']:.1%} | "
            f"60-min: {summary['recent_60min']:.1%}"
        )
    elif alert_level == "moderate":
        st.warning(
            f"**MODERATE FLARE RISK** — Maximum probability: **{max_prob:.1%}**\n\n"
            f"Current 15-min: {summary['recent_15min']:.1%} | "
            f"30-min: {summary['recent_30min']:.1%} | "
            f"60-min: {summary['recent_60min']:.1%}"
        )
    else:
        st.success(
            f"**LOW FLARE RISK** — Maximum probability: {max_prob:.1%}\n\n"
            f"Current 15-min: {summary['recent_15min']:.1%} | "
            f"30-min: {summary['recent_30min']:.1%} | "
            f"60-min: {summary['recent_60min']:.1%}"
        )


def show_flare_catalogue(cat_df):
    """Display the detected flare catalogue as a styled table."""
    if cat_df is None or cat_df.empty:
        st.info("No flare events detected.")
        return

    st.subheader("Detected Flare Events")

    display_df = cat_df[["start_time", "peak_time", "end_time", "goes_class", "peak_flux"]].copy()
    display_df["peak_flux"] = display_df["peak_flux"].apply(lambda x: f"{x:.2e}")
    display_df.columns = ["Start", "Peak", "End", "Class", "Peak Flux (W/m²)"]

    st.dataframe(display_df, use_container_width=True, hide_index=True)


def show_model_info(checkpoint_info: dict):
    """Display model information."""
    st.subheader("Model Info")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Architecture", "Parallel CNN-BiLSTM")
    with col2:
        st.metric("Parameters", "~102K")
    with col3:
        st.metric("Horizons", "15/30/60 min")

    if checkpoint_info:
        st.caption(
            f"Trained: Epoch {checkpoint_info.get('epoch', '?')} | "
            f"Val TSS: {checkpoint_info.get('val_tss', 0):.4f}"
        )
