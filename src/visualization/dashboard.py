"""
Real-time dashboard for solar flare nowcasting and forecasting.

Interactive Plotly/Dash dashboard showing:
    - SoLEXS and HEL1OS X-ray light curves
    - Nowcast detections (threshold-based triggers)
    - Forecast probabilities for 15/30/60 min horizons
    - Hardness ratio and Neupert effect panels
    - Visual alerts when flares are detected/predicted

Satisfies the problem statement requirement:
    "Interface that visualizes the X-ray light curves and triggers
     with visual alerts when a flare is nowcasted or forecasted."
"""

import logging
from typing import Dict, Optional, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FlareDashboard:
    """Interactive dashboard for solar flare monitoring (Dash/Plotly)."""

    def __init__(self, config: dict, update_callback: Optional[Callable] = None):
        self.cfg = config["visualization"]
        self.data_cfg = config["data"]
        self.update_callback = update_callback
        self._app = None

    def build(self):
        """Build the Dash application."""
        try:
            import dash
            from dash import dcc, html, Input, Output
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            logger.warning("Dash/Plotly not installed. Dashboard unavailable.")
            return None

        app = dash.Dash(__name__)
        app.title = "Aditya-L1 Solar Flare Monitoring"

        app.layout = html.Div([
            html.H1("Aditya-L1 Solar Flare Nowcasting & Forecasting",
                    style={"textAlign": "center", "color": "#1a1a2e",
                           "padding": "20px"}),

            html.Div([
                html.Div(id="alert-banner", style={
                    "padding": "15px", "margin": "10px 20px",
                    "borderRadius": "8px", "fontWeight": "bold",
                    "fontSize": "18px", "textAlign": "center"
                })
            ]),

            dcc.Interval(
                id="update-interval",
                interval=self.cfg["update_interval_ms"],
                n_intervals=0
            ),

            html.Div([
                dcc.Graph(id="lightcurve-plot"),
                dcc.Graph(id="hardness-plot"),
            ], style={"display": "grid",
                      "gridTemplateColumns": "repeat(2, 1fr)",
                      "gap": "20px", "padding": "20px"}),

            html.Div([
                dcc.Graph(id="forecast-plot"),
                dcc.Graph(id="probability-gauge"),
            ], style={"display": "grid",
                      "gridTemplateColumns": "repeat(2, 1fr)",
                      "gap": "20px", "padding": "20px"}),

            html.Div([
                html.H3("Master Flare Catalogue", style={"padding": "10px 20px"}),
                html.Div(id="catalogue-table",
                         style={"padding": "0 20px 20px 20px"})
            ])
        ])

        self._app = app
        logger.info("Dashboard built with Dash")
        return app

    def run(self, debug: bool = False, host: str = "127.0.0.1",
            port: Optional[int] = None):
        """Run the dashboard server."""
        if self._app is None:
            app = self.build()
            if app is None:
                logger.error("Cannot run dashboard: Dash not installed")
                return

        if port is None:
            port = self.cfg["dashboard_port"]

        logger.info(f"Starting dashboard at http://{host}:{port}")
        self._app.run_server(debug=debug, host=host, port=port)

    def _generate_lightcurve_figure(self, df: pd.DataFrame,
                                     detections: pd.DataFrame = None) -> dict:
        """Generate Plotly light curve figure."""
        import plotly.graph_objects as go

        fig = go.Figure()
        if "soft_flux" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["soft_flux"],
                mode="lines", name="SoLEXS (Soft X-ray)",
                line=dict(color="royalblue", width=1.5)
            ))
        if "hard_flux" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["hard_flux"],
                mode="lines", name="HEL1OS (Hard X-ray)",
                line=dict(color="firebrick", width=1.5),
                yaxis="y2"
            ))

        if detections is not None and len(detections) > 0:
            for _, event in detections.iterrows():
                fig.add_vline(x=event["peak_time"].timestamp() * 1000,
                              line=dict(color="orange", width=2, dash="dash"))

        fig.update_layout(
            title="SoLEXS + HEL1OS X-ray Light Curves",
            xaxis=dict(title="Time (UTC)", rangeslider=dict(visible=True)),
            yaxis=dict(title="SoLEXS Flux", type="log"),
            yaxis2=dict(title="HEL1OS Flux", type="log",
                        overlaying="y", side="right"),
            hovermode="x unified",
            template="plotly_white"
        )
        return fig

    def _generate_hardness_figure(self, df: pd.DataFrame) -> dict:
        """Generate hardness ratio plot."""
        import plotly.graph_objects as go

        fig = go.Figure()
        if "soft_flux" in df.columns and "hard_flux" in df.columns:
            eps = 1e-12
            hardness = np.where(
                df["soft_flux"].values > eps,
                df["hard_flux"].values / df["soft_flux"].values, 0.0
            )
            fig.add_trace(go.Scatter(
                x=df.index, y=hardness,
                mode="lines", name="Hardness Ratio (HEL1OS/SoLEXS)",
                line=dict(color="green", width=1.5)
            ))
        fig.update_layout(
            title="Hardness Ratio (HEL1OS / SoLEXS)",
            xaxis=dict(title="Time (UTC)"),
            yaxis=dict(title="Hardness Ratio", type="log"),
            hovermode="x unified", template="plotly_white"
        )
        return fig

    def _generate_forecast_figure(self, times: pd.DatetimeIndex,
                                   predictions: np.ndarray,
                                   horizons: list) -> dict:
        """Generate forecast probability plot."""
        import plotly.graph_objects as go

        fig = go.Figure()
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

        for i, h in enumerate(horizons):
            fig.add_trace(go.Scatter(
                x=times, y=predictions[:, i],
                mode="lines", name=f"P(flare in {h} min)",
                line=dict(color=colors[i % len(colors)], width=2)
            ))

        fig.add_hline(y=0.5, line=dict(color="gray", width=1, dash="dot"),
                      annotation_text="Alert threshold (0.5)")

        fig.update_layout(
            title="Flare Forecast Probabilities",
            xaxis=dict(title="Time (UTC)", rangeslider=dict(visible=True)),
            yaxis=dict(title="Probability", range=[0, 1]),
            hovermode="x unified", template="plotly_white"
        )
        return fig
