"""
Aditya-L1 Solar Flare Forecast — Streamlit App

Main entry point. Run with: streamlit run app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.app.pages.home import render_home
from src.app.pages.forecast import render_forecast_page
from src.app.pages.about import render_about
from src.app.components.sidebar import render_sidebar


def main():
    st.set_page_config(
        page_title="Aditya-L1 Solar Flare Forecast",
        page_icon="☀️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Render sidebar and get settings
    settings = render_sidebar()

    # Page navigation
    page = st.radio(
        "Navigate",
        ["Home", "Forecast", "About"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if page == "Home":
        render_home()
    elif page == "Forecast":
        render_forecast_page(settings)
    elif page == "About":
        render_about()


if __name__ == "__main__":
    main()
