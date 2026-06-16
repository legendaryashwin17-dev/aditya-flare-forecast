"""
Plotly chart components for the Streamlit app.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def plot_light_curves(
    solexs_df: pd.DataFrame,
    hel1os_df: pd.DataFrame,
    predictions: pd.DataFrame = None,
    flare_catalogue: pd.DataFrame = None,
    title: str = "Aditya-L1 X-Ray Light Curves",
) -> go.Figure:
    """Plot SoLEXS and HEL1OS light curves with dual y-axes.

    Shows soft X-ray (SoLEXS) and hard X-ray (HEL1OS) on separate axes
    with flare markers and forecast overlay.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=("X-Ray Flux", "Forecast Probability"),
    )

    # SoLEXS (soft X-ray)
    if solexs_df is not None and not solexs_df.empty:
        fig.add_trace(
            go.Scatter(
                x=solexs_df.index,
                y=solexs_df["solexs_flux"],
                name="SoLEXS (2-22 keV)",
                line=dict(color="#FF6B6B", width=1),
                hovertemplate="Time: %{x}<br>Flux: %{y:.2e} W/m²<extra>SoLEXS</extra>",
            ),
            row=1, col=1,
        )

    # HEL1OS (hard X-ray)
    if hel1os_df is not None and not hel1os_df.empty:
        fig.add_trace(
            go.Scatter(
                x=hel1os_df.index,
                y=hel1os_df["hel1os_flux"],
                name="HEL1OS (8-150 keV)",
                line=dict(color="#4ECDC4", width=1),
                yaxis="y2",
                hovertemplate="Time: %{x}<br>Flux: %{y:.2e} W/m²<extra>HEL1OS</extra>",
            ),
            row=1, col=1,
        )

    # Flare markers
    if flare_catalogue is not None and not flare_catalogue.empty:
        for _, flare in flare_catalogue.iterrows():
            fig.add_vline(
                x=flare["peak_time"],
                line_dash="dot",
                line_color="yellow",
                opacity=0.6,
                row=1, col=1,
            )
            fig.add_annotation(
                x=flare["peak_time"],
                y=0.95,
                text=flare["goes_class"],
                showarrow=False,
                font=dict(color="yellow", size=10),
                xref="x",
                yref="paper",
                row=1, col=1,
            )

    # Forecast probabilities
    if predictions is not None and not predictions.empty:
        colors = {"prob_15min": "#FF4B4B", "prob_30min": "#FFA500", "prob_60min": "#4ECDC4"}
        labels = {"prob_15min": "15 min", "prob_30min": "30 min", "prob_60min": "60 min"}

        for col, color in colors.items():
            fig.add_trace(
                go.Scatter(
                    x=predictions["timestamp"],
                    y=predictions[col],
                    name=labels[col],
                    line=dict(color=color, width=2),
                    hovertemplate=f"Time: %{{x}}<br>{labels[col]}: %{{y:.3f}}<extra></extra>",
                ),
                row=2, col=1,
            )

        # Alert threshold lines
        fig.add_hline(y=0.7, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=0.3, line_dash="dash", line_color="orange", opacity=0.5, row=2, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        height=600,
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=60, r=60, t=80, b=40),
    )

    fig.update_yaxes(title_text="Flux (W/m²)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Probability", range=[0, 1], row=2, col=1)
    fig.update_xaxes(title_text="Time (UTC)", row=2, col=1)

    return fig


def plot_forecast_bars(summary: dict) -> go.Figure:
    """Plot horizontal bar chart of forecast probabilities."""
    horizons = ["15 min", "30 min", "60 min"]
    probs = [
        summary.get("recent_15min", 0),
        summary.get("recent_30min", 0),
        summary.get("recent_60min", 0),
    ]
    max_probs = [
        summary.get("max_15min", 0),
        summary.get("max_30min", 0),
        summary.get("max_60min", 0),
    ]

    colors = []
    for p in probs:
        if p > 0.7:
            colors.append("#FF4B4B")
        elif p > 0.3:
            colors.append("#FFA500")
        else:
            colors.append("#4ECDC4")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=horizons,
        x=probs,
        name="Current",
        orientation="h",
        marker_color=colors,
        text=[f"{p:.1%}" for p in probs],
        textposition="auto",
    ))

    fig.add_trace(go.Bar(
        y=horizons,
        x=max_probs,
        name="Max",
        orientation="h",
        marker_color="rgba(255,255,255,0.2)",
        text=[f"{p:.1%}" for p in max_probs],
        textposition="auto",
    ))

    fig.update_layout(
        title="Flare Probability by Horizon",
        xaxis_title="Probability",
        yaxis_title="Forecast Horizon",
        barmode="overlay",
        template="plotly_dark",
        height=250,
        margin=dict(l=80, r=20, t=50, b=40),
        xaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def plot_hardness_ratio(solexs_df: pd.DataFrame, hel1os_df: pd.DataFrame) -> go.Figure:
    """Plot spectral hardness ratio (HEL1OS/SoLEXS) over time."""
    if solexs_df is None or hel1os_df is None:
        return go.Figure()

    # Align on common index
    common_idx = solexs_df.index.intersection(hel1os_df.index)
    if len(common_idx) == 0:
        return go.Figure()

    soft = solexs_df.loc[common_idx, "solexs_flux"].values
    hard = hel1os_df.loc[common_idx, "hel1os_flux"].values

    eps = 1e-15
    hardness = hard / np.maximum(soft, eps)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=common_idx,
        y=hardness,
        name="Hardness Ratio",
        line=dict(color="#FFD93D", width=1),
        fill="tozeroy",
        fillcolor="rgba(255,217,61,0.1)",
        hovertemplate="Time: %{x}<br>Ratio: %{y:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title="Spectral Hardness Ratio (HEL1OS / SoLEXS)",
        xaxis_title="Time (UTC)",
        yaxis_title="Hardness Ratio",
        template="plotly_dark",
        height=200,
        margin=dict(l=60, r=20, t=50, b=40),
    )

    return fig


def plot_metrics_summary(metrics: dict) -> go.Figure:
    """Plot model performance metrics as bar chart."""
    horizons = ["15 min", "30 min", "60 min"]
    tss_values = []
    auc_values = []

    for h_label in ["15min", "30min", "60min"]:
        h_data = metrics.get(h_label, {})
        tss_values.append(h_data.get("best_tss", 0))
        auc_values.append(h_data.get("auc", 0))

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="TSS",
        x=horizons,
        y=tss_values,
        marker_color="#FF6B6B",
        text=[f"{v:.3f}" for v in tss_values],
        textposition="auto",
    ))

    fig.add_trace(go.Bar(
        name="AUC",
        x=horizons,
        y=auc_values,
        marker_color="#4ECDC4",
        text=[f"{v:.3f}" for v in auc_values],
        textposition="auto",
    ))

    # Target line
    fig.add_hline(y=0.65, line_dash="dash", line_color="yellow",
                  annotation_text="Target TSS=0.65")

    fig.update_layout(
        title="Model Performance by Horizon",
        yaxis_title="Score",
        barmode="group",
        template="plotly_dark",
        height=300,
        margin=dict(l=60, r=20, t=50, b=40),
        yaxis=dict(range=[0, 1]),
    )

    return fig
