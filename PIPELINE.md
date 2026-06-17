# Aditya-L1 Solar Flare Forecast — Pipeline & Parameters Document

## Bharatiya Antariksh Hackathon 2026 — Problem Statement 15

---

## 1. Problem Statement

Build an automated algorithmic pipeline that uses combined **SoLEXS** (soft X-ray, 2-22 keV) and **HEL1OS** (hard X-ray, 8-150 keV) time-series data from **Aditya-L1** to:

- **Nowcast** (real-time detect) solar flares using threshold-based detection
- **Forecast** (predict) flares with quantifiable lead-time at 15, 30, and 60 minute horizons

**Target Metric**: TSS >= 0.65 on >=C-class flares

---

## 2. Data Sources

### Primary: Aditya-L1 (ISSDC PRADAN)

| Instrument | Energy Range | Cadence | Level | Source |
|------------|-------------|---------|-------|--------|
| SoLEXS | 2-22 keV | 1s native | L1 | PRADAN |
| HEL1OS | 8-150 keV | 1s native | L1 | PRADAN |

**Data Format**: FITS files from `https://pradan.issdc.gov.in/al1`

- SoLEXS: `AL1_SOLEXS_YYYYMMDD_SDD{1,2}_L1.lc` — HDU 1 (RATE): columns `TIME` + `COUNTS`, 86,400 rows
- HEL1OS: `lightcurve_cdte{1,2}.fits`, `lightcurve_czt{1,2}.fits` — columns `MJD`/`ISOT`/`CTR`/`STAT_ERR`, ~43,000 rows per detector

### Supplementary: GOES XRS (Transfer Learning Pre-training)

| Instrument | Time Range | Cadence | Use |
|------------|-----------|---------|-----|
| GOES XRS-A | 2003-01-01 to 2023-12-31 | 60s | Hard channel proxy |
| GOES XRS-B | 2003-01-01 to 2023-12-31 | 60s | Soft channel proxy |

---

## 3. Data Preprocessing Pipeline

### 3.1 Binning

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Native cadence | 1s | Raw instrument sampling |
| Target cadence | 10s | Reduces 86,400 to 8,640 points/hour |
| Method | Block-mean | Preserves flux statistics |

### 3.2 Gap Handling

| Parameter | Value |
|-----------|-------|
| Max gap | 300s |
| Fill method | Forward fill |

### 3.3 Feature Engineering (17 features)

#### SoLEXS Branch (7 features)

| # | Feature | Formula | Window |
|---|---------|---------|--------|
| 1 | `solexs_flux` | Raw count rate | - |
| 2 | `solexs_flux_log` | ln(flux) | - |
| 3 | `dsoft_dt_30s` | d(flux)/dt | 30s |
| 4 | `dsoft_dt_2min` | d(flux)/dt | 120s |
| 5 | `soft_rolling_mean_1min` | Rolling mean | 60s |
| 6 | `soft_rolling_std_1min` | Rolling std | 60s |
| 7 | `background_subtracted_soft` | flux - P10(flux) | - |

#### HEL1OS Branch (10 features)

| # | Feature | Formula | Window |
|---|---------|---------|--------|
| 1 | `hel1os_flux` | Raw count rate | - |
| 2 | `hel1os_flux_log` | ln(flux) | - |
| 3 | `spectral_hardness_ratio` | HEL1OS / SoLEXS | - |
| 4 | `dhard_dt_30s` | d(flux)/dt | 30s |
| 5 | `dhard_dt_2min` | d(flux)/dt | 120s |
| 6 | `overlap_xcorr` | Rolling Pearson r | 120s |
| 7 | `hard_rolling_mean_1min` | Rolling mean | 60s |
| 8 | `hard_rolling_std_1min` | Rolling std | 60s |
| 9 | `hard_rolling_mean_5min` | Rolling mean | 300s |
| 10 | `background_subtracted_hard` | flux - P10(flux) | - |

### 3.4 Standardization

- Method: Z-score (mean=0, std=1)
- Fit on training set, apply to val/test

---

## 4. Model Architecture: ParallelFlareModel

### 4.1 High-Level Architecture

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

### 4.2 Layer-by-Layer Parameters

#### SoLEXS Branch

| Layer | Type | In | Out | Kernel | Stride | Padding |
|-------|------|----|-----|--------|--------|---------|
| solexs_conv.0 | Conv1D | 7 | 32 | 3 | 1 | same |
| solexs_conv.1 | BatchNorm1D | 32 | 32 | - | - | - |
| solexs_conv.2 | ReLU | - | - | - | - | - |

#### HEL1OS Branch

| Layer | Type | In | Out | Kernel | Stride | Padding |
|-------|------|----|-----|--------|--------|---------|
| hel1os_conv.0 | Conv1D | 10 | 32 | 3 | 1 | same |
| hel1os_conv.1 | BatchNorm1D | 32 | 32 | - | - | - |
| hel1os_conv.2 | ReLU | - | - | - | - | - |

#### Overlap Convolution

| Layer | Type | In | Out | Kernel | Stride | Padding |
|-------|------|----|-----|--------|--------|---------|
| overlap_conv.0 | Conv1D | 64 | 64 | 5 | 1 | same |
| overlap_conv.1 | BatchNorm1D | 64 | 64 | - | - | - |
| overlap_conv.2 | ReLU | - | - | - | - | - |

#### BiLSTM

| Parameter | Value |
|-----------|-------|
| Input size | 64 |
| Hidden size | 64 |
| Layers | 1 |
| Bidirectional | True |
| Dropout | 0.0 |
| Output size | 128 (64 * 2) |

#### Output Heads (3 independent)

| Layer | Type | In | Out |
|-------|------|----|-----|
| head.0 | Linear | 128 | 32 |
| head.1 | ReLU | - | - |
| head.2 | Dropout(0.4) | - | - |
| head.3 | Linear | 32 | 1 |

### 4.3 Total Parameters

```
Total:     101,539
Trainable: 101,539
```

### 4.4 Weight Initialization

| Layer Type | Method |
|------------|--------|
| Conv1D / Linear | Kaiming normal (fan_out, relu) |
| LSTM weights | Orthogonal |
| LSTM bias | Zeros |

---

## 5. Training Configuration

### 5.1 Loss Function: Focal Loss

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| alpha | 0.75 | Upweight positive class |
| gamma | 2.0 | Focus on hard examples |
| eps | 1e-7 | Numerical stability |

### 5.2 Optimizer

| Parameter | Value |
|-----------|-------|
| Algorithm | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |

### 5.3 Learning Rate Schedule

| Parameter | Value |
|-----------|-------|
| Scheduler | ReduceLROnPlateau |
| Mode | max (monitor TSS) |
| Factor | 0.5 |
| Patience | 3 epochs |
| Min LR | 1e-6 |

### 5.4 Regularization

| Technique | Parameter | Value |
|-----------|-----------|-------|
| Dropout | heads | 0.4 |
| Weight decay | L2 | 1e-4 |
| Gradient clipping | max_norm | 1.0 |
| Early stopping | patience | 10 epochs |
| Early stopping | metric | val_tss |

### 5.5 Data Handling

| Parameter | Value |
|-----------|-------|
| Batch size | 32 |
| Max epochs | 100 |
| Oversampling | Enabled (5x) |
| Train/Val/Test split | 80% / 10% / 10% |
| Split method | Chronological |

### 5.6 Transfer Learning (Planned)

| Parameter | Value |
|-----------|-------|
| Pre-train data | GOES XRS (2003-2023) |
| Freeze branches | SoLEXS + HEL1OS CNN |
| Fine-tune LR | 1e-4 |
| Trainable layers | Overlap Conv + BiLSTM + Heads |

---

## 6. Input / Output Specification

### 6.1 Input

| Parameter | Value |
|-----------|-------|
| Shape | [B, 360, 17] |
| Window | 60 minutes |
| Cadence | 10 seconds |
| Steps | 360 |
| Stride | 30 steps (5 minutes) |
| Features | 17 (7 SoLEXS + 10 HEL1OS) |

### 6.2 Output

| Parameter | Value |
|-----------|-------|
| Shape | [B, 3] |
| Format | Sigmoid probabilities |
| Horizons | 15 min, 30 min, 60 min |
| Range | [0, 1] |
| Interpretation | P(C+ class flare within horizon) |

---

## 7. Evaluation Metrics

### 7.1 Primary Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| TSS (True Skill Statistic) | POD - POFD | >= 0.65 |
| HSS (Heidke Skill Score) | (H-E)/(K-E) | Maximize |
| AUC-ROC | Area under ROC curve | Maximize |

### 7.2 Operational Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| POD (Probability of Detection) | TP/(TP+FN) | Missed flare rate |
| POFD (Prob of False Detection) | FP/(FP+TN) | False alarm rate |
| FAR (False Alarm Ratio) | FP/(TP+FP) | Precision complement |
| Brier Score | mean((p-y)^2) | Calibration |

### 7.3 Evaluation Thresholds

| Threshold | Description |
|-----------|-------------|
| 0.1 | Low (high sensitivity) |
| 0.2 | Moderate |
| 0.3 | Balanced |
| 0.5 | High specificity |
| 0.7 | Very high specificity |

---

## 8. Training Results

### 8.1 Overall

| Metric | Value |
|--------|-------|
| Best Val TSS | 0.331 |
| Test Mean TSS | 0.438 |
| Test Mean AUC | 0.757 |
| Total Parameters | 101,539 |
| Training Flares | 30 |

### 8.2 Per-Horizon Results

#### 15-Minute Horizon

| Threshold | TSS | HSS | POD | POFD | FAR | Precision |
|-----------|-----|-----|-----|------|-----|-----------|
| 0.3 | 0.347 | 0.038 | 1.000 | 0.653 | 0.944 | 0.056 |
| **0.5** | **0.750** | **0.852** | **0.750** | **0.000** | **0.000** | **1.000** |
| 0.7 | 0.750 | 0.852 | 0.750 | 0.000 | 0.000 | 1.000 |

- **Best TSS: 0.750** (threshold 0.5)
- **AUC: 0.909**
- Brier: 0.097

#### 30-Minute Horizon

| Threshold | TSS | HSS | POD | POFD | FAR | Precision |
|-----------|-----|-----|-----|------|-----|-----------|
| 0.3 | 0.010 | 0.002 | 1.000 | 0.990 | 0.924 | 0.076 |
| 0.5 | 0.375 | 0.526 | 0.375 | 0.000 | 0.000 | 1.000 |
| 0.7 | 0.375 | 0.526 | 0.375 | 0.000 | 0.000 | 1.000 |

- **Best TSS: 0.375** (threshold 0.5)
- **AUC: 0.741**
- Brier: 0.196

#### 60-Minute Horizon

| Threshold | TSS | HSS | POD | POFD | FAR | Precision |
|-----------|-----|-----|-----|------|-----|-----------|
| 0.3 | 0.000 | 0.000 | 1.000 | 1.000 | 0.850 | 0.150 |
| 0.5 | 0.004 | 0.001 | 1.000 | 0.996 | 0.850 | 0.150 |
| 0.7 | 0.188 | 0.282 | 0.188 | 0.000 | 0.000 | 1.000 |

- **Best TSS: 0.188** (threshold 0.7)
- **AUC: 0.621**
- Brier: 0.286

---

## 9. Nowcasting (Threshold-Based Detection)

| Parameter | Value |
|-----------|-------|
| Soft channel threshold | 3.0 sigma |
| Hard channel threshold | 3.0 sigma |
| dF/dt threshold | 2.0 sigma |
| Consecutive samples | 3 |
| Cooldown | 15 minutes |
| Merge window | 60 seconds |

---

## 10. Key Innovation: 8-22 keV Overlap Band

The model focuses on the **8-22 keV overlap** between SoLEXS and HEL1OS, where:

- **Pre-flare precursor brightening** appears 15-25 minutes before flare onset (Nandi et al. 2025)
- **Spectral Hardness Ratio** (HEL1OS/SoLEXS) captures thermal-to-non-thermal transition
- **Rolling Pearson Cross-Correlation** detects correlated emission across instruments

Two features specifically target this band:
1. `spectral_hardness_ratio` — ratio of hard/soft flux
2. `overlap_xcorr` — Pearson correlation in rolling window

---

## 11. References

1. Nandi et al. 2025 — HEL1OS pre-flare precursor brightening in 8-22 keV band
2. Sarwade et al. 2025 — SoLEXS instrument design and calibration
3. Hassani et al. 2025 — TSS=0.74 benchmark on GOES XRS (2003-2023)
4. Lin et al. 2017 — Focal Loss for Dense Object Detection (ICCV)
5. Bloomfield et al. 2012 — TSS/HSS standards for space weather (ApJ 747, 41)
6. Neupert Effect — d(soft X-ray)/dt tracks hard X-ray flux

---

## 12. Configuration File

All parameters are stored in `config/config.yaml`. The training script reads from this file.

```yaml
# Full configuration path
config/config.yaml
```

---

*Document generated: 2026-06-16*
*Project: Aditya-L1 Solar Flare Forecast*
*Hackathon: Bharatiya Antariksh 2026 — Problem Statement 15*
