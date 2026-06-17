"""
Simulated flare data generator for demo purposes.

Generates realistic SoLEXS + HEL1OS light curves with:
- Quiet-Sun baseline
- Flare events with impulsive rise and decay
- Pre-flare precursor brightening (Nandi et al. 2025)
- Multiple flare classes (A, B, C, M, X)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple


# GOES class peak flux boundaries (W/m^2)
CLASS_PEAK_FLUXES = {
    "A": 1e-8,
    "B": 1e-7,
    "C": 1e-6,
    "M": 1e-5,
    "X": 1e-4,
}


def _flare_profile(
    duration_min: float,
    peak_flux: float,
    cadence_s: float = 10.0,
    include_precursor: bool = True,
    precursor_duration_min: float = 20.0,
    precursor_fraction: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a single flare profile with rise and decay phases.

    Returns:
        (time_s, flux) arrays
    """
    n_points = int(duration_min * 60 / cadence_s)
    t = np.arange(n_points) * cadence_s

    # Rise phase: ~25% of duration
    rise_end = int(n_points * 0.25)
    # Decay phase: ~75% of duration
    decay_start = rise_end

    flux = np.zeros(n_points)

    # Impulsive rise (exponential)
    for i in range(rise_end):
        frac = i / max(rise_end, 1)
        flux[i] = peak_flux * (1 - np.exp(-3 * frac))

    # Peak
    flux[rise_end] = peak_flux

    # Gradual decay (exponential)
    decay_len = n_points - decay_start
    for i in range(decay_len):
        frac = i / max(decay_len, 1)
        flux[decay_start + i] = peak_flux * np.exp(-2.5 * frac)

    # Add pre-flare precursor brightening (Nandi et al. 2025)
    if include_precursor:
        precursor_points = int(precursor_duration_min * 60 / cadence_s)
        precursor_flux = peak_flux * precursor_fraction
        precursor_start = max(0, rise_end - precursor_points)
        for i in range(precursor_start, rise_end):
            progress = (i - precursor_start) / max(rise_end - precursor_start, 1)
            # Gradual increase with accelerating slope
            flux[i] += precursor_flux * (progress ** 2)

    return t, flux


def generate_simulated_data(
    duration_hours: float = 24.0,
    n_flares: int = 3,
    cadence_s: float = 10.0,
    seed: int = None,
    start_time: datetime = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate simulated SoLEXS + HEL1OS light curves with flares.

    Returns:
        (solexs_df, hel1os_df, flare_catalogue)
        Each DataFrame has DatetimeIndex and flux column.
        Flare catalogue has columns: start_time, peak_time, end_time, goes_class, peak_flux
    """
    if seed is None:
        seed = int(np.random.randint(0, 100000))
    rng = np.random.RandomState(seed)

    if start_time is None:
        start_time = datetime(2024, 10, 15, 0, 0, 0)

    n_points = int(duration_hours * 3600 / cadence_s)
    times = [start_time + timedelta(seconds=i * cadence_s) for i in range(n_points)]

    # Quiet-Sun baseline (slightly variable)
    quiet_solexs = 1e-7 + 1e-9 * rng.randn(n_points).cumsum()  # ~1e-7 W/m^2
    quiet_hel1os = 1e-9 + 1e-11 * rng.randn(n_points).cumsum()  # ~1e-9 W/m^2

    # Add slow drift
    t_hours = np.arange(n_points) * cadence_s / 3600
    quiet_solexs += 1e-8 * np.sin(2 * np.pi * t_hours / 12)  # 12-hour cycle
    quiet_hel1os += 1e-10 * np.sin(2 * np.pi * t_hours / 8)  # 8-hour cycle

    solexs_flux = quiet_solexs.copy()
    hel1os_flux = quiet_hel1os.copy()

    flare_catalogue = []

    # Place flares at random times
    if n_flares > 0:
        flare_times = sorted(rng.choice(
            range(int(n_points * 0.1), int(n_points * 0.9)),
            size=min(n_flares, n_points // 100),
            replace=False
        ))

        for ft in flare_times:
            # Random flare class (weighted toward C and B)
            class_choice = rng.choice(["B", "B", "C", "C", "C", "M", "X"],
                                       p=[0.15, 0.15, 0.25, 0.25, 0.1, 0.08, 0.02])
            peak_flux = CLASS_PEAK_FLUXES[class_choice] * (0.5 + rng.random())

            # Duration: 5-30 minutes
            duration = rng.uniform(5, 30)

            t_f, flux_f = _flare_profile(
                duration, peak_flux, cadence_s,
                include_precursor=True,
                precursor_duration_min=rng.uniform(10, 25),
                precursor_fraction=rng.uniform(0.1, 0.3)
            )

            # Add to light curves
            end_idx = min(ft + len(flux_f), n_points)
            actual_len = end_idx - ft

            # SoLEXS: softer spectrum (2-22 keV)
            solexs_flux[ft:end_idx] += flux_f[:actual_len] * (1.0 + 0.2 * rng.random())

            # HEL1OS: harder spectrum (8-150 keV), stronger impulsive phase
            hel1os_flux[ft:end_idx] += flux_f[:actual_len] * 0.3 * (1.0 + 0.3 * rng.random())

            peak_idx = ft + int(actual_len * 0.25)
            peak_time = times[min(peak_idx, n_points - 1)]

            flare_catalogue.append({
                "start_time": times[ft],
                "peak_time": peak_time,
                "end_time": times[end_idx - 1],
                "goes_class": class_choice,
                "peak_flux": float(peak_flux),
            })

    solexs_df = pd.DataFrame(
        {"solexs_flux": solexs_flux},
        index=pd.DatetimeIndex(times, name="time")
    )
    hel1os_df = pd.DataFrame(
        {"hel1os_flux": hel1os_flux},
        index=pd.DatetimeIndex(times, name="time")
    )
    cat_df = pd.DataFrame(flare_catalogue) if flare_catalogue else pd.DataFrame(
        columns=["start_time", "peak_time", "end_time", "goes_class", "peak_flux"]
    )

    return solexs_df, hel1os_df, cat_df
