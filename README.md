# Aditya-L1 Solar Flare Forecasting & Nowcasting

**Bharatiya Antariksh Hackathon 2026 -- Problem Statement 15**

Automated pipeline for nowcasting (real-time detection) and forecasting (predictive) of solar flares using combined soft X-ray (SoLEXS) and hard X-ray (HEL1OS) data from ISRO's Aditya-L1 mission.

## PRIMARY Dataset: Aditya-L1 SoLEXS + HEL1OS

| Instrument | Energy Range | Cadence | Detectors | Data Access |
|---|---|---|---|---|
| **SoLEXS** (Soft X-ray) | 2-22 keV | 1s spectral, 0.1s timing | 2x SDD (SDD1: quiet Sun, SDD2: flares) | ISSDC PRADAN |
| **HEL1OS** (Hard X-ray) | 8-150 keV | 10ms events, 1s light curves, 20s PHA | 2x CdTe (8-70 keV) + 2x CZT (20-150 keV) | ISSDC PRADAN |

**Physics rationale for joint SoLEXS+HEL1OS (Nandi et al. 2025, Sarwade et al. 2025):**
- **Precursor band (8-22 keV overlap):** Pre-flare brightening appears near 20 keV -- exactly the SoLEXS-HEL1OS spectral overlap region
- **Neupert effect:** d(SoLEXS flux)/dt correlates with HEL1OS flux (non-thermal electron driver vs thermal response)
- **Hardness ratio:** HEL1OS/SoLEXS ratio tracks spectral hardening before flare onset (NOAA operational method)
- **Full spectral coverage:** 2-150 keV spans thermal plasma (soft) through non-thermal particle acceleration (hard)

## Supplementary: GOES XRS + HEK

GOES XRS-A (0.5-4 A, hard proxy) and XRS-B (1-8 A, soft proxy) provide 20+ years of training augmentation (2003-2023, 151,071 flares -- Hassani et al. 2025). The same physics (hardness ratio, Neupert effect) transfers directly.

## Architecture: CNN-BiLSTM-Attention

Total Parameters: ~1.1M (lightweight, ~5MB checkpoint)

Input: [T x F] 60-min multi-channel (SoLEXS flux, HEL1OS flux, hardness ratio, derivatives, rolling stats)

1D CNN (3 layers, filters=[64,128,256], kernel=5, ReLU, BatchNorm, MaxPool) -> ~200K params
BiLSTM (2 layers, hidden=128, dropout=0.3, bidirectional) -> ~530K params
Temporal Attention -> ~16K params
Dense (64->32->3) -> ~350K params
Sigmoid -> [P(15min), P(30min), P(60min)]

## Training Heuristics

| Hyperparameter | Value | Source / Rationale |
|---|---|---|
| CNN layers | 3 | Tang et al. 2020, arXiv:2002.10061 |
| CNN kernel size | 5 | Optimal for 1D time series classification |
| LSTM layers | 2 | Hassani et al. 2025, TSS=0.74 benchmark |
| LSTM hidden size | 128 | Balance of capacity vs overfitting |
| Dropout | 0.3 | LSTM regularization standard |
| Focal Loss alpha | 0.75 | Down-weight easy negatives (Lin et al. 2017) |
| Focal Loss gamma | 2.0 | Focus on hard misclassified examples |
| Optimizer | Adam (lr=1e-4, wd=1e-5) | Stable convergence |
| Batch size | 64 | Memory x throughput tradeoff |
| Max epochs | 100 | Early stopping patience=15 |
| Input window | 60 min | Precursor timescale |
| Forecast horizons | 15/30/60 min | Lead-time curve (problem requirement) |
| Train/val/test | 80/10/10 chronological | No temporal leakage |
| Oversample factor | 5x | Mitigate class imbalance |

## Evaluation Metrics

| Metric | Definition | Target | Literature Benchmark |
|---|---|---|---|
| TSS | POD - POFD | >=0.65 | 0.74 (Hassani et al. 2025, GOES) |
| AUC-ROC | Area under ROC | >=0.80 | 0.87 (Hassani et al. 2025, GOES) |
| FAR | FP / (TP + FP) | <0.30 | Problem requirement |
| Lead Time | Min before flare peak | >10 min | Problem requirement |
| HSS | Heidke Skill Score | >0.50 | Bloomfield et al. 2012 |
| POD (Recall) | TP / (TP + FN) | >=0.80 | -- |

## Quick Start

`ash
pip install -r requirements.txt
python main.py --mode pipeline                    # Full pipeline (dev mode with simulated data)
python main.py --mode pipeline --solexs path/to/solexs.fits --hel1os path/to/hel1os.fits
python main.py --mode train --solexs solexs.fits --hel1os hel1os.fits
python main.py --mode evaluate --checkpoint data/models/best_model.pt
python main.py --mode dashboard
python -m pytest tests/ -v
`

## Data Access

ISSDC PRADAN Portal: https://pradan.issdc.gov.in

1. Register at PRADAN and request SoLEXS + HEL1OS access
2. Download Level-1 light curve FITS files
3. Point the pipeline to your files via --solexs and --hel1os flags

## Project Structure

`
aditya-flare-forecast/
  config/config.yaml           # All hyperparameters
  src/
    data/ingestion.py          # Aditya-L1 SoLEXS+HEL1OS FITS reader (PRIMARY)
    data/preprocessing.py      # Physics-informed feature engineering
    data/dataset.py            # PyTorch sliding window dataset
    models/cnn_bilstm_attention.py  # ~1.1M param model
    models/focal_loss.py            # Focal loss (Lin et al. 2017)
    nowcasting/                # Threshold detection + catalogue merger
    evaluation/metrics.py      # TSS, HSS, FAR, AUC, lead time
    visualization/             # Dash dashboard + static plots
    train.py                   # Training loop with early stopping
    pipeline.py                # End-to-end orchestrator
  main.py                      # CLI entry point
  tests/                       # 19 passing tests
  requirements.txt
  README.md
`

## References

1. Nandi et al. 2025, "HEL1OS -- A Hard X-ray Spectrometer on Board Aditya-L1", arXiv:2512.12679
2. Sarwade et al. 2025, "Solar Low Energy X-ray Spectrometer on board Aditya-L1", arXiv:2509.26292
3. Hassani et al. 2025, "Solar Flare Prediction Using LSTM and Decomposition-LSTM", ApJS, 279, 27
4. Lin et al. 2017, "Focal Loss for Dense Object Detection", ICCV, arXiv:1708.02002
5. Tang et al. 2020, "Rethinking 1D-CNN for Time Series Classification", arXiv:2002.10061
6. Bloomfield et al. 2012, "Toward a Benchmarking Standard for Solar Flare Forecasting", ApJ, 747, 41
