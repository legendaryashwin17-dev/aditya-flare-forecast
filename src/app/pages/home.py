"""Premium home page — landing page with hero, stats, pipeline."""

import streamlit as st


def render_home():
    # Noise overlay
    st.markdown('<div class="noise-overlay"></div>', unsafe_allow_html=True)

    # Hero Section
    st.markdown("""
    <div class="hero-container">
        <div class="hero-eyebrow">
            <span class="pulse-dot"></span>
            Aditya-L1 Mission • SoLEXS + HEL1OS
        </div>
        <h1 class="hero-title">
            Solar Flare<br>
            <span class="gradient-text">Forecast System</span>
        </h1>
        <p class="hero-subtitle">
            Real-time nowcasting and multi-horizon forecasting of solar flares
            using X-ray time-series from India's first solar observatory.
            Powered by parallel CNN-BiLSTM with 8-22 keV overlap optimization.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Stats Grid
    st.markdown("""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-label">Forecast Horizon</div>
            <div class="stat-value accent">15 / 30 / 60</div>
            <div class="stat-desc">minutes ahead</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Instruments</div>
            <div class="stat-value">2</div>
            <div class="stat-desc">SoLEXS + HEL1OS</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Architecture</div>
            <div class="stat-value">~102K</div>
            <div class="stat-desc">parameters</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Best TSS</div>
            <div class="stat-value accent">0.75</div>
            <div class="stat-desc">@ 15 min horizon</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Best AUC</div>
            <div class="stat-value accent">0.91</div>
            <div class="stat-desc">area under curve</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Features</div>
            <div class="stat-value">17</div>
            <div class="stat-desc">physics-informed</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline Section
    st.markdown("""
    <div class="section-header stagger-3">
        <div class="section-eyebrow">Pipeline</div>
        <h2 class="section-title">How It Works</h2>
        <p class="section-desc">
            From raw FITS telemetry to probabilistic flare forecasts in four steps.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="pipeline-grid">
        <div class="pipeline-step stagger-3">
            <div class="step-number">01</div>
            <div class="step-title">Data Ingestion</div>
            <div class="step-desc">
                SoLEXS (2-22 keV) and HEL1OS (8-150 keV) Level-1 FITS files
                from ISSDC PRADAN portal. 1s native cadence.
            </div>
        </div>
        <div class="pipeline-step stagger-4">
            <div class="step-number">02</div>
            <div class="step-title">Preprocessing</div>
            <div class="step-desc">
                10s binning, 17 physics-informed features: spectral hardness ratio,
                flux derivatives, rolling cross-correlation, background subtraction.
            </div>
        </div>
        <div class="pipeline-step stagger-5">
            <div class="step-number">03</div>
            <div class="step-title">Model Inference</div>
            <div class="step-desc">
                Parallel 1D-CNN branches with overlap Conv, BiLSTM(64),
                three independent output heads for each forecast horizon.
            </div>
        </div>
        <div class="pipeline-step stagger-6">
            <div class="step-number">04</div>
            <div class="step-title">Forecast Output</div>
            <div class="step-desc">
                Probability of C+ class flare at 15, 30, and 60 minutes.
                Color-coded alerts with quantified confidence.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Innovation Section
    st.markdown("""
    <div class="section-header" style="margin-top:3rem;">
        <div class="section-eyebrow">Key Innovation</div>
        <h2 class="section-title">8-22 keV Overlap Band</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="dbez stagger-4" style="max-width:800px;">
        <div class="dbez-inner">
            <p style="color:var(--text-secondary);line-height:1.75;font-size:0.95rem;">
                The model focuses on the <strong style="color:var(--accent);">8-22 keV overlap</strong>
                between SoLEXS and HEL1OS, where <strong style="color:var(--text-primary);">pre-flare
                precursor brightening</strong> has been observed (Nandi et al. 2025). This band provides
                predictive information <strong style="color:var(--text-primary);">15-25 minutes before
                flare onset</strong>, enabling the model to achieve TSS=0.75 at the 15-minute horizon.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Getting Started
    st.markdown("""
    <div class="section-header" style="margin-top:3rem;">
        <div class="section-eyebrow">Get Started</div>
        <h2 class="section-title">Try It Now</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="pipeline-grid" style="grid-template-columns:1fr 1fr;">
        <div class="pipeline-step stagger-5">
            <div class="step-number">A</div>
            <div class="step-title">Simulated Data</div>
            <div class="step-desc">
                Click "Forecast" in the sidebar and select "Simulated Flares"
                to see the model analyze synthetic flare events with pre-flare precursors.
            </div>
        </div>
        <div class="pipeline-step stagger-6">
            <div class="step-number">B</div>
            <div class="step-title">Real Data</div>
            <div class="step-desc">
                Download FITS files from
                <a href="https://pradan.issdc.gov.in/al1" target="_blank"
                   style="color:var(--accent);text-decoration:none;">ISSDC PRADAN</a>
                or use the auto-download feature in the sidebar.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:3rem;padding:1.5rem;background:var(--bg-card);
        border:1px solid var(--border-subtle);border-radius:var(--radius-lg);
        text-align:center;">
        <span style="font-family:'Geist Mono',monospace;font-size:0.7rem;
            letter-spacing:0.15em;text-transform:uppercase;color:var(--accent);">
            Bharatiya Antariksh Hackathon 2026 — Problem Statement 15
        </span>
    </div>
    """, unsafe_allow_html=True)
