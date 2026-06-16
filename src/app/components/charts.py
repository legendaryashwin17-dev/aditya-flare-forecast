"""Premium Plotly chart components with dark glass aesthetic."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

COLORS = {
    "solexs": "#ff6b6b",
    "hel1os": "#00ffaa",
    "prob_15": "#ff4757",
    "prob_30": "#ffa502",
    "prob_60": "#00ffaa",
    "hardness": "#ffd93d",
    "grid": "rgba(255,255,255,0.04)",
    "text": "#7a7a8a",
    "bg": "rgba(0,0,0,0)",
}


def _base_layout(height=600, **kwargs):
    return dict(
        height=height,
        template="plotly_dark",
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(family="Space Grotesk, sans-serif", color=COLORS["text"], size=12),
        margin=dict(l=60, r=40, t=60, b=50),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11, color=COLORS["text"]),
        ),
        xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        **kwargs,
    )


def plot_light_curves(solexs_df, hel1os_df, predictions=None,
                      flare_catalogue=None, title="Aditya-L1 X-Ray Light Curves"):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=("X-Ray Flux", "Forecast Probability"),
    )

    if solexs_df is not None and not solexs_df.empty:
        fig.add_trace(go.Scatter(
            x=solexs_df.index, y=solexs_df["solexs_flux"],
            name="SoLEXS (2-22 keV)",
            line=dict(color=COLORS["solexs"], width=1),
            hovertemplate="Time: %{x}<br>Flux: %{y:.2e} W/m2<extra>SoLEXS</extra>",
        ), row=1, col=1)

    if hel1os_df is not None and not hel1os_df.empty:
        fig.add_trace(go.Scatter(
            x=hel1os_df.index, y=hel1os_df["hel1os_flux"],
            name="HEL1OS (8-150 keV)",
            line=dict(color=COLORS["hel1os"], width=1),
            yaxis="y2",
            hovertemplate="Time: %{x}<br>Flux: %{y:.2e} W/m2<extra>HEL1OS</extra>",
        ), row=1, col=1)

    if flare_catalogue is not None and not flare_catalogue.empty:
        for _, flare in flare_catalogue.iterrows():
            fig.add_vline(x=flare["peak_time"], line_dash="dot",
                          line_color="#ffd93d", opacity=0.5, row=1, col=1)
            fig.add_annotation(
                x=flare["peak_time"], y=0.95, text=flare["goes_class"],
                showarrow=False, font=dict(color="#ffd93d", size=10),
                xref="x", yref="paper", row=1, col=1,
            )

    if predictions is not None and not predictions.empty:
        colors = {"prob_15min": COLORS["prob_15"], "prob_30min": COLORS["prob_30"],
                  "prob_60min": COLORS["prob_60"]}
        labels = {"prob_15min": "15 min", "prob_30min": "30 min", "prob_60min": "60 min"}
        for col, color in colors.items():
            fig.add_trace(go.Scatter(
                x=predictions["timestamp"], y=predictions[col],
                name=labels[col], line=dict(color=color, width=2),
                hovertemplate=f"Time: %{{x}}<br>{labels[col]}: %{{y:.3f}}<extra></extra>",
            ), row=2, col=1)
        fig.add_hline(y=0.7, line_dash="dash", line_color="rgba(255,71,87,0.4)",
                      row=2, col=1)
        fig.add_hline(y=0.3, line_dash="dash", line_color="rgba(255,165,2,0.3)",
                      row=2, col=1)

    fig.update_layout(**_base_layout(
        height=600,
        title=dict(text=title, font=dict(size=14, color="#e8e8ec")),
    ))
    fig.update_yaxes(title_text="Flux (W/m2)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Probability", range=[0, 1], row=2, col=1)
    fig.update_xaxes(title_text="Time (UTC)", row=2, col=1)
    return fig


def plot_forecast_bars(summary: dict):
    horizons = ["15 min", "30 min", "60 min"]
    probs = [summary.get("recent_15min", 0), summary.get("recent_30min", 0),
             summary.get("recent_60min", 0)]
    max_probs = [summary.get("max_15min", 0), summary.get("max_30min", 0),
                 summary.get("max_60min", 0)]

    colors = [COLORS["prob_15"] if p > 0.7 else (COLORS["prob_30"] if p > 0.3 else COLORS["prob_60"])
              for p in probs]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=horizons, x=probs, name="Current", orientation="h",
        marker_color=colors, text=[f"{p:.1%}" for p in probs], textposition="auto",
    ))
    fig.add_trace(go.Bar(
        y=horizons, x=max_probs, name="Max", orientation="h",
        marker_color="rgba(255,255,255,0.12)", text=[f"{p:.1%}" for p in max_probs],
        textposition="auto",
    ))
    fig.update_layout(**_base_layout(
        height=250,
        title=dict(text="Flare Probability by Horizon", font=dict(size=13, color="#e8e8ec")),
        barmode="overlay", xaxis=dict(range=[0, 1]),
    ))
    return fig


def plot_hardness_ratio(solexs_df, hel1os_df):
    if solexs_df is None or hel1os_df is None:
        return go.Figure()

    common_idx = solexs_df.index.intersection(hel1os_df.index)
    if len(common_idx) == 0:
        return go.Figure()

    soft = solexs_df.loc[common_idx, "solexs_flux"].values
    hard = hel1os_df.loc[common_idx, "hel1os_flux"].values
    eps = 1e-15
    hardness = hard / np.maximum(soft, eps)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=common_idx, y=hardness, name="Hardness Ratio",
        line=dict(color=COLORS["hardness"], width=1),
        fill="tozeroy", fillcolor="rgba(255,217,61,0.08)",
        hovertemplate="Time: %{x}<br>Ratio: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        height=200,
        title=dict(text="Spectral Hardness Ratio (HEL1OS / SoLEXS)",
                   font=dict(size=13, color="#e8e8ec")),
        xaxis_title="Time (UTC)", yaxis_title="Hardness Ratio",
    ))
    return fig
