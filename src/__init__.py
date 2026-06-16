"""
Aditya-L1 Solar Flare Forecasting & Nowcasting Pipeline.

Combines SoLEXS (soft X-ray, 2-22 keV) and HEL1OS (hard X-ray, 8-150 keV)
data from ISRO's Aditya-L1 mission for automated flare detection and prediction.

References:
    - Hassani et al. 2025, ApJS, 279, 27 (LSTM/DLSTM for flare prediction)
    - Nandi et al. 2025, arXiv:2512.12679 (HEL1OS instrument)
    - Sarwade et al. 2025, arXiv:2509.26292 (SoLEXS instrument)
    - Lin et al. 2017, ICCV (Focal Loss for class imbalance)
    - Tang et al. 2020, arXiv:2002.10061 (1D-CNN for time series)
"""
