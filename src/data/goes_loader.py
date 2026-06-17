"""
GOES XRS data loader and flare catalogue for transfer learning.

Downloads GOES XRS data from NOAA and creates flare catalogues
for pre-training and transfer learning to Aditya-L1.

Data source: NOAA SWPC GOES XRS
- XRS-A: 0.5-4 Angstrom (hard channel proxy)
- XRS-B: 1-8 Angstrom (soft channel proxy)
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, List
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# GOES class peak flux thresholds (W/m^2)
GOES_THRESHOLDS = {
    "A": 1e-8,
    "B": 1e-7,
    "C": 1e-6,
    "M": 1e-5,
    "X": 1e-4,
}


def fetch_goes_xrs(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    cache_dir: str = "data/goes",
) -> pd.DataFrame:
    """Fetch GOES XRS data from NOAA SWPC.

    Returns DataFrame with columns: time, xrs_a, xrs_b
    """
    try:
        import goes2xrs
    except ImportError:
        logger.info("goes2xrs not installed, using manual download")
        return _fetch_goes_manual(start_date, end_date, cache_dir)

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    logger.info(f"Fetching GOES XRS: {start_date} to {end_date}")

    dfs = []
    current = start
    while current <= end:
        year = current.year
        month = current.month
        try:
            df = goes2xrs.fetch(year, month)
            if df is not None and len(df) > 0:
                dfs.append(df)
                logger.info(f"  {year}-{month:02d}: {len(df)} rows")
        except Exception as e:
            logger.warning(f"  {year}-{month:02d}: failed ({e})")
        current += timedelta(days=32)
        current = current.replace(day=1)

    if not dfs:
        logger.warning("No GOES data fetched")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("time").reset_index(drop=True)

    logger.info(f"Total GOES XRS: {len(combined)} rows")
    return combined


def _fetch_goes_manual(
    start_date: str,
    end_date: str,
    cache_dir: str,
) -> pd.DataFrame:
    """Manual GOES XRS download as fallback."""
    import urllib.request
    import json

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Use NOAA SWPC JSON API
    base_url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"

    logger.info(f"Fetching GOES XRS from NOAA SWPC...")
    try:
        req = urllib.request.Request(base_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())

        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["time_tag"])
        df = df.rename(columns={"flux": "xrs_a", "flux_b": "xrs_b"})

        # Keep relevant columns
        cols = ["time"]
        for c in ["xrs_a", "xrs_b"]:
            if c in df.columns:
                cols.append(c)
        df = df[cols]

        logger.info(f"Fetched {len(df)} rows from NOAA SWPC")
        return df
    except Exception as e:
        logger.warning(f"Failed to fetch GOES data: {e}")
        return pd.DataFrame()


def load_goes_from_file(filepath: str) -> pd.DataFrame:
    """Load GOES XRS data from a local CSV/FITS file."""
    filepath = Path(filepath)

    if filepath.suffix == ".csv":
        df = pd.read_csv(filepath, parse_dates=["time"])
    elif filepath.suffix in (".fits", ".fit"):
        from astropy.io import fits
        with fits.open(filepath) as hdulist:
            for hdu in hdulist[1:]:
                if hdu.data is not None:
                    df = pd.DataFrame(hdu.data)
                    if "TIME" in df.columns:
                        from astropy.time import Time
                        times = Time(df["TIME"], format="mjd").to_datetime()
                        df["time"] = times
                    break
    else:
        raise ValueError(f"Unsupported format: {filepath.suffix}")

    return df


def create_goes_flare_catalogue(
    goes_df: pd.DataFrame,
    min_class: str = "C",
) -> pd.DataFrame:
    """Create flare catalogue from GOES XRS data.

    Identifies flares as contiguous regions above background threshold.

    Returns DataFrame with columns:
        start_time, peak_time, end_time, goes_class, peak_flux
    """
    if goes_df.empty:
        return pd.DataFrame(columns=["start_time", "peak_time", "end_time",
                                     "goes_class", "peak_flux"])

    # Use XRS-B (1-8 A) for flare identification
    flux_col = "xrs_b" if "xrs_b" in goes_df.columns else "xrs_a"
    flux = goes_df[flux_col].values
    times = goes_df["time"].values

    # Background: 10th percentile
    valid_flux = flux[flux > 0]
    if len(valid_flux) == 0:
        return pd.DataFrame(columns=["start_time", "peak_time", "end_time",
                                     "goes_class", "peak_flux"])

    background = np.percentile(valid_flux, 10)
    threshold = max(background * 3, GOES_THRESHOLDS[min_class])

    # Find flare regions (contiguous above threshold)
    above = flux > threshold
    flares = []
    in_flare = False
    start_idx = 0

    for i in range(len(above)):
        if above[i] and not in_flare:
            in_flare = True
            start_idx = i
        elif not above[i] and in_flare:
            in_flare = False
            end_idx = i - 1
            peak_idx = start_idx + np.argmax(flux[start_idx:end_idx + 1])

            peak_flux = float(flux[peak_idx])
            goes_class = _flux_to_class(peak_flux)

            flares.append({
                "start_time": pd.Timestamp(times[start_idx]),
                "peak_time": pd.Timestamp(times[peak_idx]),
                "end_time": pd.Timestamp(times[end_idx]),
                "goes_class": goes_class,
                "peak_flux": peak_flux,
            })

    catalogue = pd.DataFrame(flares)
    logger.info(f"GOES flare catalogue: {len(catalogue)} flares "
                f"(min class: {min_class})")

    return catalogue


def _flux_to_class(peak_flux: float) -> str:
    """Convert peak flux to GOES class letter."""
    if peak_flux >= GOES_THRESHOLDS["X"]:
        return "X"
    elif peak_flux >= GOES_THRESHOLDS["M"]:
        return "M"
    elif peak_flux >= GOES_THRESHOLDS["C"]:
        return "C"
    elif peak_flux >= GOES_THRESHOLDS["B"]:
        return "B"
    else:
        return "A"


def align_goes_to_aditya(
    goes_df: pd.DataFrame,
    aditya_times: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Resample GOES data to match Aditya-L1 timestamps.

    Uses linear interpolation to align GOES 1-minute cadence
    to Aditya-L1 10-second cadence.
    """
    if goes_df.empty:
        return goes_df

    goes_df = goes_df.set_index("time").sort_index()

    # Resample to target cadence
    resampled = goes_df.reindex(aditya_times, method="ffill")

    # Interpolate missing values
    resampled = resampled.interpolate(method="time", limit=60)

    return resampled.reset_index().rename(columns={"index": "time"})


def prepare_goes_for_transfer(
    goes_df: pd.DataFrame,
    aditya_times: pd.DatetimeIndex,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare GOES data as proxy for SoLEXS/HEL1OS branches.

    Returns:
        (solexs_proxy, hel1os_proxy) arrays aligned to aditya_times
    """
    aligned = align_goes_to_aditya(goes_df, aditya_times)

    if aligned.empty:
        return np.array([]), np.array([])

    # XRS-B -> SoLEXS proxy (soft X-rays)
    solexs_proxy = aligned["xrs_b"].values if "xrs_b" in aligned.columns else aligned["xrs_a"].values

    # XRS-A -> HEL1OS proxy (harder X-rays)
    hel1os_proxy = aligned["xrs_a"].values if "xrs_a" in aligned.columns else aligned["xrs_b"].values

    return solexs_proxy, hel1os_proxy


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test GOES data fetch
    goes_df = fetch_goes_xrs("2024-01-01", "2024-03-01")
    if not goes_df.empty:
        print(f"GOES data: {len(goes_df)} rows")
        print(f"Columns: {goes_df.columns.tolist()}")

        # Create flare catalogue
        cat = create_goes_flare_catalogue(goes_df, min_class="C")
        print(f"Flares: {len(cat)}")
        if not cat.empty:
            print(cat.head())
