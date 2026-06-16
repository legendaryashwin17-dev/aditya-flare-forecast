"""
FITS reader for Aditya-L1 SoLEXS and HEL1OS Level-1 data from PRADAN.

PRADAN data formats:
    SoLEXS: AL1_SOLEXS_YYYYMMDD_SDD{1,2}_L1.lc
        HDU 1 (RATE): columns TIME (float64, elapsed seconds) + COUNTS (float64)
        86,400 rows = 24h at 1s cadence

    HEL1OS: lightcurve_cdte{1,2}.fits, lightcurve_czt{1,2}.fits
        Multiple HDUs per energy band (EXTNAME like CDTE1_LC_BAND_5.00KEV_TO_20.00KEV)
        Columns: MJD (float64), ISOT (string), CTR (float64), STAT_ERR (float64)
        ~43,000 rows per band

    HEL1OS also has: evt.fits (event list), spectra files, GTI files
"""

import logging
import gzip
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def detect_instrument(filepath: str) -> str:
    """Detect whether a FITS file is SoLEXS or HEL1OS based on filename."""
    name = Path(filepath).stem.lower()
    if "solexs" in name or "sdd" in name or "slx" in name:
        return "solexs"
    if "hel1os" in name or "cdte" in name or "czt" in name or "hls" in name:
        return "hel1os"
    return "unknown"


def extract_zip(zip_path: str, extract_dir: str) -> Path:
    """Extract a PRADAN ZIP file and decompress .gz files.

    Returns the directory containing FITS files.
    """
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting {zip_path.name}...")

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    # Decompress .gz files
    for gz_file in extract_dir.rglob("*.gz"):
        out_file = gz_file.with_suffix("")
        if not out_file.exists():
            logger.info(f"  Decompressing: {gz_file.name}")
            with gzip.open(gz_file, "rb") as f_in:
                with open(out_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

    return extract_dir


def read_solexs_lc(filepath: str) -> pd.DataFrame:
    """Read a SoLEXS light curve FITS file.

    Format: AL1_SOLEXS_YYYYMMDD_SDD{1,2}_L1.lc
    HDU 1 (RATE): TIME (elapsed seconds) + COUNTS
    """
    from astropy.io import fits

    hdulist = fits.open(filepath)
    logger.info(f"Opened SoLEXS: {filepath} ({len(hdulist)} HDUs)")

    # Find the RATE extension
    time_data = None
    counts_data = None

    for hdu in hdulist:
        if hdu.name == "RATE" and hdu.data is not None:
            col_names = [c.upper() for c in hdu.data.dtype.names]
            if "TIME" in col_names and "COUNTS" in col_names:
                time_data = np.array(hdu.data["TIME"], dtype=np.float64)
                counts_data = np.array(hdu.data["COUNTS"], dtype=np.float64)
                logger.info(f"  Found RATE extension: {len(time_data)} rows")
                break

    # Fallback: try any extension with TIME+COUNTS
    if time_data is None:
        for i, hdu in enumerate(hdulist):
            if hdu.data is None or not hasattr(hdu.data, "dtype"):
                continue
            if hdu.data.dtype.names is None:
                continue
            col_names = [c.upper() for c in hdu.data.dtype.names]
            if "TIME" in col_names and "COUNTS" in col_names:
                time_data = np.array(hdu.data["TIME"], dtype=np.float64)
                counts_data = np.array(hdu.data["COUNTS"], dtype=np.float64)
                logger.info(f"  Fallback HDU[{i}]: {len(time_data)} rows")
                break

    hdulist.close()

    if time_data is None or counts_data is None:
        raise ValueError(f"Could not find TIME+COUNTS in {filepath}")

    # Read MJDREFI from header for absolute time
    mjdrefi = None
    with fits.open(filepath) as hdulist:
        for hdu in hdulist:
            if hasattr(hdu, "header") and "MJDREFI" in hdu.header:
                mjdrefi = hdu.header["MJDREFI"]
                mjdreff = hdu.header.get("MJDREFF", 0.0)
                mjdrefi = mjdrefi + mjdreff
                break

    # Convert elapsed seconds to datetime
    if mjdrefi is not None:
        from astropy.time import Time
        mjd = mjdrefi + time_data / 86400.0
        times = Time(mjd, format="mjd").to_datetime()
    else:
        from datetime import datetime, timedelta
        base = datetime(2024, 7, 1)
        times = [base + timedelta(seconds=t) for t in time_data]

    # Clean flux
    counts_data = np.where(np.isfinite(counts_data), counts_data, 0.0)
    counts_data = np.maximum(counts_data, 0.0)

    df = pd.DataFrame(
        {"solexs_flux": counts_data},
        index=pd.DatetimeIndex(times, name="time"),
    )
    df = df.sort_index()

    logger.info(f"  Read {len(df)} samples, "
                f"flux: [{df['solexs_flux'].min():.1f}, {df['solexs_flux'].max():.1f}]")
    return df


def read_hel1os_lc(filepath: str, band: str = "full") -> pd.DataFrame:
    """Read a HEL1OS light curve FITS file.

    Format: lightcurve_cdte{1,2}.fits or lightcurve_czt{1,2}.fits
    Multiple HDUs per energy band with columns: MJD, ISOT, CTR, STAT_ERR

    Band options: 'full', '5-20', '20-30', '30-40', '40-60',
                  '20-40', '60-80', '80-150'
    """
    from astropy.io import fits

    hdulist = fits.open(filepath)
    logger.info(f"Opened HEL1OS: {filepath} ({len(hdulist)} HDUs)")

    # Find the appropriate band HDU
    target_hdu = None

    if band == "full":
        # Use the widest band (last HDU typically)
        for hdu in hdulist[1:]:
            if hdu.data is not None and len(hdu.data) > 100:
                extname = hdu.name.upper()
                if "BAND" in extname:
                    target_hdu = hdu
                    # Prefer wider bands
                    if "1.80KEV" in extname or "18.00KEV" in extname:
                        break
    else:
        # Find specific band
        for hdu in hdulist[1:]:
            if hdu.data is not None:
                extname = hdu.name.upper()
                if band.replace("-", ".") in extname.replace("-", "."):
                    target_hdu = hdu
                    break

    # Fallback: use first data extension
    if target_hdu is None:
        for hdu in hdulist[1:]:
            if hdu.data is not None and len(hdu.data) > 100:
                target_hdu = hdu
                break

    if target_hdu is None or target_hdu.data is None:
        hdulist.close()
        raise ValueError(f"Could not find light curve data in {filepath}")

    # Read data BEFORE closing the file
    extname = target_hdu.name
    data = target_hdu.data.copy()
    col_names = [c.upper() for c in data.dtype.names]
    n_rows = len(data)
    logger.info(f"  Using band: {extname} ({n_rows} rows)")

    hdulist.close()

    # Time: try MJD first, then ISOT
    time_data = None
    if "MJD" in col_names:
        time_data = np.array(data["MJD"], dtype=np.float64)
        from astropy.time import Time
        times = Time(time_data, format="mjd").to_datetime()
    elif "ISOT" in col_names:
        from astropy.time import Time
        isot_strs = [s.decode() if isinstance(s, bytes) else str(s) for s in data["ISOT"]]
        times = Time(isot_strs, format="isot").to_datetime()
    elif "TIME" in col_names:
        time_data = np.array(data["TIME"], dtype=np.float64)
        from datetime import datetime, timedelta
        base = datetime(2024, 7, 1)
        times = [base + timedelta(seconds=t) for t in time_data]
    else:
        raise ValueError(f"No time column found in {filepath}")

    # Flux: CTR (count rate)
    if "CTR" in col_names:
        flux = np.array(data["CTR"], dtype=np.float64)
    elif "COUNTS" in col_names:
        flux = np.array(data["COUNTS"], dtype=np.float64)
    elif "RATE" in col_names:
        flux = np.array(data["RATE"], dtype=np.float64)
    else:
        raise ValueError(f"No flux column found in {filepath}")

    # Clean
    flux = np.where(np.isfinite(flux), flux, 0.0)
    flux = np.maximum(flux, 0.0)

    df = pd.DataFrame(
        {"hel1os_flux": flux},
        index=pd.DatetimeIndex(times, name="time"),
    )
    df = df.sort_index()

    logger.info(f"  Read {len(df)} samples, "
                f"flux: [{df['hel1os_flux'].min():.2f}, {df['hel1os_flux'].max():.2f}]")
    return df


def read_hel1os_combined(
    cdte1_path: Optional[str] = None,
    cdte2_path: Optional[str] = None,
    czt1_path: Optional[str] = None,
    czt2_path: Optional[str] = None,
    band: str = "full",
) -> pd.DataFrame:
    """Read and combine HEL1OS light curves from multiple detectors.

    Sums count rates from all available detectors for the requested band.
    """
    dfs = []

    for path, name in [(cdte1_path, "CdTe1"), (cdte2_path, "CdTe2"),
                       (czt1_path, "CZT1"), (czt2_path, "CZT2")]:
        if path:
            try:
                df = read_hel1os_lc(path, band=band)
                dfs.append(df)
                logger.info(f"  Added {name}: {len(df)} samples")
            except Exception as e:
                logger.warning(f"  Failed to read {name}: {e}")

    if not dfs:
        raise ValueError("No HEL1OS data could be read")

    # Combine by summing overlapping time bins
    combined = dfs[0].copy()
    for df in dfs[1:]:
        combined = combined.add(df, fill_value=0.0)

    combined = combined.sort_index()
    logger.info(f"Combined HEL1OS: {len(combined)} samples from {len(dfs)} detectors")
    return combined


def read_fits_lightcurve(
    filepath: str,
    instrument: Optional[str] = None,
) -> pd.DataFrame:
    """Read a FITS file and extract time + flux as a DataFrame.

    Auto-detects SoLEXS vs HEL1OS format.
    """
    if instrument is None:
        instrument = detect_instrument(filepath)

    if instrument == "solexs":
        return read_solexs_lc(filepath)
    elif instrument == "hel1os":
        return read_hel1os_lc(filepath, band="full")
    else:
        raise ValueError(f"Unknown instrument for {filepath}")


def read_pair(
    solexs_path: Optional[str],
    hel1os_path: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read SoLEXS and HEL1OS FITS files and return as a pair.

    Returns:
        (solexs_df, hel1os_df) — each with DatetimeIndex and flux column.
    """
    solexs_df = None
    hel1os_df = None

    if solexs_path:
        logger.info(f"Reading SoLEXS: {solexs_path}")
        solexs_df = read_fits_lightcurve(solexs_path, instrument="solexs")

    if hel1os_path:
        logger.info(f"Reading HEL1OS: {hel1os_path}")
        hel1os_df = read_fits_lightcurve(hel1os_path, instrument="hel1os")

    return solexs_df, hel1os_df


def find_fits_in_dir(directory: str) -> Tuple[Optional[str], Optional[str]]:
    """Find SoLEXS and HEL1OS FITS files in a directory.

    Searches for known PRADAN filename patterns.

    Returns:
        (solexs_path, hel1os_path)
    """
    directory = Path(directory)
    solexs_path = None
    hel1os_path = None

    # SoLEXS: look for .lc files or SDD files
    for pattern in ["*SDD*_L1.lc", "*solexs*.lc", "*SLX*.lc"]:
        matches = list(directory.rglob(pattern))
        if matches:
            solexs_path = str(matches[0])
            break

    # HEL1OS: look for lightcurve files
    for pattern in ["lightcurve_czt1.fits", "lightcurve_cdte1.fits",
                    "*hel1os*lightcurve*", "*HEL1OS*LC*"]:
        matches = list(directory.rglob(pattern))
        if matches:
            hel1os_path = str(matches[0])
            break

    return solexs_path, hel1os_path
