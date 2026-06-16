"""Premium about page — architecture, metrics, references."""

import streamlit as st
from pathlib import Path
import json


def render_about():
    st.markdown('<div class="noise-overlay"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:2rem 0 1rem 0;">
        <div class="section-eyebrow">Documentation</div>
        <h2 class="section-title" style="font-size:clamp(1.5rem,3vw,2.2rem);">About This Project</h2>
    </div>
    """, unsafe_allow_html=True)

    # Problem Statement
    st.markdown("""
    <div class="section-header">
        <div class="section-eyebrow">Problem Statement</div>
        <h3 class="section-title" style="font-size:1.3rem;">Automated Solar Flare Forecasting from Aditya-L1</h3>
    </div>

    <div class="dbez" style="max-width:900px;">
        <div class="dbez-inner">
            <p style="color:var(--text-secondary);line-height:1.75;font-size:0.95rem;">
                Build an automated algorithmic pipeline that uses combined
                <strong style="color:var(--accent);">SoLEXS</strong> (soft X-ray, 2-22 keV) and
                <strong style="color:var(--accent);">HEL1OS</strong> (hard X-ray, 8-150 keV)
                time-series data from <strong style="color:var(--text-primary);">Aditya-L1</strong> to:
            </p>
            <ul style="color:var(--text-secondary);line-height:2;padding-left:1.5rem;margin-top:0.75rem;">
                <li><strong style="color:var(--text-primary);">Nowcast</strong> solar flares using threshold-based detection</li>
                <li><strong style="color:var(--text-primary);">Forecast</strong> flares with quantifiable lead-time (15/30/60 min)</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Architecture
    st.markdown("""
    <div class="section-header" style="margin-top:3rem;">
        <div class="section-eyebrow">Architecture</div>
        <h3 class="section-title" style="font-size:1.3rem;">Parallel 1D-CNN + BiLSTM + Multi-Head</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="arch-diagram">
<span class="arch-highlight">Input:</span> [B, 360, 17]  <span class="arch-comment">(60-min window at 10s cadence, 17 features)</span>
     |
+----+----+
|         |
<span class="arch-highlight">SoLEXS</span>   <span class="arch-highlight">HEL1OS</span>        <span class="arch-comment">&lt;- Parallel branches (sensor-specific noise)</span>
|         |
1D-Conv   1D-Conv
(32,k=3)  (32,k=3)
|         |
+----+----+
     |
<span class="arch-highlight">Overlap Conv</span>          <span class="arch-comment">&lt;- 64 filters, k=5 (8-22 keV cross-correlation)</span>
     |
<span class="arch-highlight">BiLSTM (64)</span>           <span class="arch-comment">&lt;- Single layer, bidirectional</span>
     |
+----+----+
|    |    |
Head1 Head2 Head3      <span class="arch-comment">&lt;- Multi-head: 15, 30, 60 min</span>
Dense(32) -&gt; Drop(0.4) -&gt; Dense(1, sigmoid)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'Geist Mono',monospace;font-size:0.75rem;
        color:var(--text-muted);margin-top:0.75rem;">
        Parameters: ~102,000 (lightweight, fast inference)
    </div>
    """, unsafe_allow_html=True)

    # Features Table
    st.markdown("""
    <div class="section-header" style="margin-top:3rem;">
        <div class="section-eyebrow">Feature Engineering</div>
        <h3 class="section-title" style="font-size:1.3rem;">17 Physics-Informed Features</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <table class="feature-table">
        <thead>
            <tr>
                <th>Branch</th>
                <th>Features</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong style="color:var(--text-primary);">SoLEXS (7)</strong></td>
                <td>
                    <span class="feat-tag">flux</span>
                    <span class="feat-tag">log(flux)</span>
                    <span class="feat-tag">dF/dt 30s</span>
                    <span class="feat-tag">dF/dt 2min</span>
                    <span class="feat-tag">rolling mean 1min</span>
                    <span class="feat-tag">rolling std 1min</span>
                    <span class="feat-tag">bg-subtracted</span>
                </td>
            </tr>
            <tr>
                <td><strong style="color:var(--text-primary);">HEL1OS (10)</strong></td>
                <td>
                    <span class="feat-tag">flux</span>
                    <span class="feat-tag">log(flux)</span>
                    <span class="feat-tag">hardness ratio</span>
                    <span class="feat-tag">dF/dt 30s</span>
                    <span class="feat-tag">dF/dt 2min</span>
                    <span class="feat-tag">overlap xcorr</span>
                    <span class="feat-tag">rolling mean 1min</span>
                    <span class="feat-tag">rolling std 1min</span>
                    <span class="feat-tag">rolling mean 5min</span>
                    <span class="feat-tag">bg-subtracted</span>
                </td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    # Training Details
    st.markdown("""
    <div class="section-header" style="margin-top:3rem;">
        <div class="section-eyebrow">Training</div>
        <h3 class="section-title" style="font-size:1.3rem;">Hyperparameters</h3>
    </div>

    <div class="pipeline-grid" style="grid-template-columns:repeat(auto-fit,minmax(180px,1fr));">
        <div class="pipeline-step">
            <div class="step-title" style="font-size:0.9rem;">Loss Function</div>
            <div class="step-desc">Focal Loss (alpha=0.75, gamma=2.0)</div>
        </div>
        <div class="pipeline-step">
            <div class="step-title" style="font-size:0.9rem;">Optimizer</div>
            <div class="step-desc">Adam (lr=1e-3, weight_decay=1e-4)</div>
        </div>
        <div class="pipeline-step">
            <div class="step-title" style="font-size:0.9rem;">LR Schedule</div>
            <div class="step-desc">ReduceLROnPlateau (factor=0.5, patience=3)</div>
        </div>
        <div class="pipeline-step">
            <div class="step-title" style="font-size:0.9rem;">Early Stopping</div>
            <div class="step-desc">Monitor val TSS, patience=10</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Training Results
    results_path = Path(__file__).parent.parent.parent.parent / "data" / "training_results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

        st.markdown("""
        <div class="section-header" style="margin-top:3rem;">
            <div class="section-eyebrow">Results</div>
            <h3 class="section-title" style="font-size:1.3rem;">Training Performance</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-item">
                <div class="metric-val">{results.get('test_mean_tss', 0):.4f}</div>
                <div class="metric-label">Mean TSS</div>
            </div>
            <div class="metric-item">
                <div class="metric-val">{results.get('test_mean_auc', 0):.4f}</div>
                <div class="metric-label">Mean AUC</div>
            </div>
            <div class="metric-item">
                <div class="metric-val">{results.get('best_val_tss', 0):.4f}</div>
                <div class="metric-label">Best Val TSS</div>
            </div>
            <div class="metric-item">
                <div class="metric-val">{results.get('n_params', 0):,}</div>
                <div class="metric-label">Parameters</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if "per_horizon" in results:
            for horizon, data in results["per_horizon"].items():
                tss = data.get("best_tss", 0)
                auc = data.get("auc", 0)
                bar_width = int(tss * 100)
                color = "safe" if tss >= 0.65 else ("warning" if tss >= 0.3 else "danger")
                st.markdown(f"""
                <div style="margin:1rem 0;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:0.35rem;">
                        <span style="font-family:'Geist Mono',monospace;font-size:0.75rem;
                            color:var(--text-secondary);">{horizon}</span>
                        <span style="font-family:'Geist Mono',monospace;font-size:0.75rem;
                            color:var(--text-primary);">TSS={tss:.4f} | AUC={auc:.4f}</span>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill {color}" style="width:{bar_width}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # References
    st.markdown("""
    <div class="section-header" style="margin-top:3rem;">
        <div class="section-eyebrow">References</div>
        <h3 class="section-title" style="font-size:1.3rem;">Key Publications</h3>
    </div>

    <div style="display:flex;flex-direction:column;gap:0.75rem;max-width:800px;">
        <div style="padding:1rem 1.25rem;background:var(--bg-card);border:1px solid var(--border-subtle);
            border-radius:var(--radius-lg);">
            <span style="font-family:'Geist Mono',monospace;font-size:0.65rem;color:var(--accent);">01</span>
            <span style="font-size:0.9rem;color:var(--text-secondary);margin-left:0.75rem;">
                <strong style="color:var(--text-primary);">Nandi et al. 2025</strong> —
                HEL1OS pre-flare precursor brightening in 8-22 keV band
            </span>
        </div>
        <div style="padding:1rem 1.25rem;background:var(--bg-card);border:1px solid var(--border-subtle);
            border-radius:var(--radius-lg);">
            <span style="font-family:'Geist Mono',monospace;font-size:0.65rem;color:var(--accent);">02</span>
            <span style="font-size:0.9rem;color:var(--text-secondary);margin-left:0.75rem;">
                <strong style="color:var(--text-primary);">Sarwade et al. 2025</strong> —
                SoLEXS instrument design and calibration
            </span>
        </div>
        <div style="padding:1rem 1.25rem;background:var(--bg-card);border:1px solid var(--border-subtle);
            border-radius:var(--radius-lg);">
            <span style="font-family:'Geist Mono',monospace;font-size:0.65rem;color:var(--accent);">03</span>
            <span style="font-size:0.9rem;color:var(--text-secondary);margin-left:0.75rem;">
                <strong style="color:var(--text-primary);">Hassani et al. 2025</strong> —
                TSS=0.74 benchmark on GOES XRS (2003-2023)
            </span>
        </div>
        <div style="padding:1rem 1.25rem;background:var(--bg-card);border:1px solid var(--border-subtle);
            border-radius:var(--radius-lg);">
            <span style="font-family:'Geist Mono',monospace;font-size:0.65rem;color:var(--accent);">04</span>
            <span style="font-size:0.9rem;color:var(--text-secondary);margin-left:0.75rem;">
                <strong style="color:var(--text-primary);">Lin et al. 2017</strong> —
                Focal Loss for Dense Object Detection
            </span>
        </div>
        <div style="padding:1rem 1.25rem;background:var(--bg-card);border:1px solid var(--border-subtle);
            border-radius:var(--radius-lg);">
            <span style="font-family:'Geist Mono',monospace;font-size:0.65rem;color:var(--accent);">05</span>
            <span style="font-size:0.9rem;color:var(--text-secondary);margin-left:0.75rem;">
                <strong style="color:var(--text-primary);">Neupert Effect</strong> —
                d(soft X-ray)/dt tracks hard X-ray flux
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Data Source
    st.markdown("""
    <div style="margin-top:3rem;padding:1.5rem;background:var(--bg-card);
        border:1px solid var(--border-subtle);border-radius:var(--radius-lg);text-align:center;">
        <span style="font-family:'Geist Mono',monospace;font-size:0.65rem;letter-spacing:0.1em;
            text-transform:uppercase;color:var(--text-muted);">Data Source</span><br>
        <a href="https://pradan.issdc.gov.in/al1" target="_blank"
           style="color:var(--accent);text-decoration:none;font-size:0.9rem;">
            ISSDC PRADAN Portal — Aditya-L1 Science Data
        </a>
    </div>
    """, unsafe_allow_html=True)
