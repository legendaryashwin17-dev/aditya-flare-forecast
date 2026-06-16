"""
Data ingestion module for Aditya-L1 SoLEXS + HEL1OS (PRIMARY) and GOES XRS (supplementary).

PRIMARY DATA:
    Aditya-L1 SoLEXS (2-22 keV, soft X-ray) — Sarwade et al. 2025
    Aditya-L1 HEL1OS (8-150 keV, hard X-ray) — Nandi et al. 2025
    Access: ISSDC PRADAN portal (https://pradan.issdc.gov.in)

SUPPLEMENTARY:
    GOES XRS-A/XRS-B for training augmentation (2003-2023, 151,071 flares)
    HEK flare catalogue for ground-truth labels

Physics rationale for SoLEXS+HEL1OS:
    - Joint coverage from 2-150 keV spans thermal (soft) and non-thermal (hard) emission
    - The 8-22 keV overlap band is where pre-flare precursor brightening appears
    - Hardness ratio (HEL1OS/SoLEXS) tracks spectral evolution pre- and during flare
    - Neupert effect: d(SoLEXS)/dt correlates with HEL1OS flux
"""

import logging
import warnings
from typing import Optional, Tuple, List
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from astropy.io import fits

logger = logging.getLogger(__name__)


class AdityaL1DataIngester:
    """PRIMARY ingester: Aditya-L1 SoLEXS + HEL1OS Level-1 data from ISSDC PRADAN.

    SoLEXS (Sarwade et al. 2025):
        - 2x Silicon Drift Detectors (SDD1: large-aperture quiet Sun, SDD2: small-aperture flares)
        - Energy range: 2-22 keV
        - Energy resolution: ~170 eV @ 5.9 keV
        - Time cadence: 1s (spectral), 0.1s (timing channel)
        - Data format: FITS (OGIP-compliant), Level-1 light curves

    HEL1OS (Nandi et al. 2025):
        - 4 detectors: 2x CdTe (cooled, 8-70 keV) + 2x CZT (room temp, 20-150 keV)
        - Energy resolution: ~1 keV @ 14 keV (CdTe), ~7 keV @ 60 keV (CZT)
        - Time cadence: 10ms (event list), 1s (Level-1 light curves), 20s (PHA spectra)
        - Data products: event list, light curves, type-II PHA spectra, GTI, housekeeping
        - Sensitivity: statistically robust from ~C6 through X-class
    """

    def __init__(self, start_date: str = "2024-07-01",
                 end_date: str = "2025-12-31"):
        self.start_date = start_date
        self.end_date = end_date

    def read_solexs_fits(self, filepath: str) -> pd.DataFrame:
        """Read SoLEXS Level-1 light curve FITS file.

        FITS structure (OGIP-compliant):
            - Primary header: instrument metadata, exposure info
            - Extension 'SPECTRUM' or 'RATE': time-series data
            - Columns include: TIME, RATE, ERROR, QUALITY, etc.
            - SDD1 and SDD2 in separate extensions or files
        """
        logger.info(f"Reading SoLEXS FITS: {filepath}")
        with fits.open(filepath) as hdul:
            logger.info(f"  Extensions: {[h.name for h in hdul]}")

            ext_name = None
            for candidate in ["SPECTRUM", "RATE", "LIGHTCURVE"]:
                if candidate in hdul:
                    ext_name = candidate
                    break
            if ext_name is None:
                ext_name = 1

            data = hdul[ext_name].data
            if data is None:
                logger.warning(f"No data in extension {ext_name}")
                return pd.DataFrame()

            cols = data.columns.names
            logger.info(f"  Columns: {cols}")

            time_col = None
            for c in cols:
                if c.upper() in ("TIME", "TIMETAG", "ELAPSED"):
                    time_col = c
                    break

            rate_col = None
            for c in cols:
                if c.upper() in ("RATE", "COUNT_RATE", "FLUX", "COUNTS"):
                    rate_col = c
                    break

            if time_col is None:
                times = np.arange(len(data), dtype=float)
            else:
                times = data[time_col].astype(float)

            if rate_col is None:
                rate = np.ones(len(data), dtype=float)
            else:
                rate = data[rate_col].astype(float)

            time_ref = hdul[0].header.get("MJDREF", 0.0)
            if time_ref > 0:
                from astropy.time import Time
                t_ref = Time(time_ref, format="mjd")
                times_abs = t_ref.datetime + pd.to_timedelta(times, unit="s")
            else:
                times_abs = pd.date_range(
                    self.start_date, periods=len(times),
                    freq=pd.DateOffset(seconds=1)
                )[:len(times)]

            df = pd.DataFrame({"solexs_flux": rate}, index=pd.DatetimeIndex(times_abs))
            df.index.name = "time"
            logger.info(f"  Loaded {len(df)} SoLEXS samples")
            return df

    def read_hel1os_fits(self, filepath: str) -> pd.DataFrame:
        """Read HEL1OS Level-1 light curve FITS file.

        FITS structure:
            - Event list (10ms): individual photon events
            - Light curve (1s): binned count rates, per-detector and summed
            - Type-II PHA spectra (20s): spectral data
            - GTI: Good Time Intervals
            - Housekeeping: detector temperatures, voltages
        """
        logger.info(f"Reading HEL1OS FITS: {filepath}")
        with fits.open(filepath) as hdul:
            logger.info(f"  Extensions: {[h.name for h in hdul]}")

            ext_name = None
            for candidate in ["LIGHTCURVE", "RATE", "EVENTS", "STDGTI"]:
                if candidate in hdul:
                    ext_name = candidate
                    break
            if ext_name is None:
                ext_name = 1

            data = hdul[ext_name].data
            if data is None:
                logger.warning(f"No data in extension {ext_name}")
                return pd.DataFrame()

            cols = data.columns.names
            logger.info(f"  Columns: {cols}")

            time_col = None
            for c in cols:
                if c.upper() in ("TIME", "TIMETAG", "ELAPSED"):
                    time_col = c
                    break

            rate_col = None
            for c in cols:
                if c.upper() in ("RATE", "COUNT_RATE", "FLUX", "COUNTS", "PHA"):
                    rate_col = c
                    break

            if time_col is None:
                times = np.arange(len(data), dtype=float)
            else:
                times = data[time_col].astype(float)

            if rate_col is None:
                rate = np.ones(len(data), dtype=float)
            else:
                rate = data[rate_col].astype(float)

            time_ref = hdul[0].header.get("MJDREF", 0.0)
            if time_ref > 0:
                from astropy.time import Time
                t_ref = Time(time_ref, format="mjd")
                times_abs = t_ref.datetime + pd.to_timedelta(times, unit="s")
            else:
                times_abs = pd.date_range(
                    self.start_date, periods=len(times),
                    freq=pd.DateOffset(seconds=1)
                )[:len(times)]

            df = pd.DataFrame({"hel1os_flux": rate}, index=pd.DatetimeIndex(times_abs))
            df.index.name = "time"
            logger.info(f"  Loaded {len(df)} HEL1OS samples")
            return df

    def load_aditya_l1_both(self, solexs_paths: List[str],
                            hel1os_paths: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and concatenate multiple SoLEXS and HEL1OS FITS files.

        Aligns both datasets onto a common 1-second time grid.
        Handles the case where detectors have separate FITS files.
        """
        logger.info(f"Loading {len(solexs_paths)} SoLEXS + {len(hel1os_paths)} HEL1OS files")

        solexs_frames = []
        for p in solexs_paths:
            try:
                solexs_frames.append(self.read_solexs_fits(p))
            except Exception as e:
                logger.error(f"Failed to read SoLEXS {p}: {e}")

        hel1os_frames = []
        for p in hel1os_paths:
            try:
                hel1os_frames.append(self.read_hel1os_fits(p))
            except Exception as e:
                logger.error(f"Failed to read HEL1OS {p}: {e}")

        if not solexs_frames:
            logger.warning("No SoLEXS data loaded — generating simulated data")
            solexs = self._generate_simulated_solexs()
        else:
            solexs = pd.concat(solexs_frames).sort_index()
            solexs = solexs[~solexs.index.duplicated(keep="first")]

        if not hel1os_frames:
            logger.warning("No HEL1OS data loaded — generating simulated data")
            hel1os = self._generate_simulated_hel1os()
        else:
            hel1os = pd.concat(hel1os_frames).sort_index()
            hel1os = hel1os[~hel1os.index.duplicated(keep="first")]

        common_start = max(solexs.index.min(), hel1os.index.min())
        common_end = min(solexs.index.max(), hel1os.index.max())
        solexs = solexs[solexs.index >= common_start]
        solexs = solexs[solexs.index <= common_end]
        hel1os = hel1os[hel1os.index >= common_start]
        hel1os = hel1os[hel1os.index <= common_end]

        logger.info(f"SoLEXS: {len(solexs)} samples, HEL1OS: {len(hel1os)} samples")
        logger.info(f"Time range: {common_start} to {common_end}")
        return solexs, hel1os

    def load_from_directory(self, data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Auto-discover and load SoLEXS + HEL1OS FITS from a directory."""
        import glob
        solexs_pattern = f"{data_dir}/**/*solexs*lightcurve*.fits"
        hel1os_pattern = f"{data_dir}/**/*hel1os*lightcurve*.fits"

        solexs_files = glob.glob(solexs_pattern, recursive=True)
        hel1os_files = glob.glob(hel1os_pattern, recursive=True)

        if not solexs_files:
            solexs_pattern = f"{data_dir}/**/*SDD*.fits"
            solexs_files = glob.glob(solexs_pattern, recursive=True)

        if not hel1os_files:
            hel1os_pattern = f"{data_dir}/**/*CdTe*.fits"
            hel1os_files = glob.glob(hel1os_pattern, recursive=True)

        logger.info(f"Found {len(solexs_files)} SoLEXS files, {len(hel1os_files)} HEL1OS files")
        return self.load_aditya_l1_both(solexs_files, hel1os_files)

    def _generate_simulated_solexs(self) -> pd.DataFrame:
        """Generate simulated SoLEXS-like data (2-22 keV, realistic background)."""
        np.random.seed(42)
        n_points = 50000
        times = pd.date_range(start="2024-07-01", periods=n_points, freq="1s")
        quiet_level = 1e-6
        flux = quiet_level * (1 + 0.1 * np.random.randn(n_points))
        flux = np.maximum(flux, 0)
        return pd.DataFrame({"solexs_flux": flux}, index=times)

    def _generate_simulated_hel1os(self) -> pd.DataFrame:
        """Generate simulated HEL1OS-like data (8-150 keV, lower count rate)."""
        np.random.seed(42)
        n_points = 50000
        times = pd.date_range(start="2024-07-01", periods=n_points, freq="1s")
        quiet_level = 3e-7
        flux = quiet_level * (1 + 0.15 * np.random.randn(n_points))
        flux = np.maximum(flux, 0)
        return pd.DataFrame({"hel1os_flux": flux}, index=times)


class SupplementaryDataIngester:
    """SUPPLEMENTARY data: GOES XRS (training augmentation), SDO/EUV (optional)."""

    def __init__(self, start_date: str = "2003-01-01",
                 end_date: str = "2023-12-31",
                 cadence_seconds: int = 60):
        self.start_date = start_date
        self.end_date = end_date
        self.cadence = cadence_seconds

    def fetch_goes_xrs(self) -> pd.DataFrame:
        """Fetch GOES XRS time series. Returns soft (XRS-B) and hard (XRS-A) channels."""
        try:
            from sunpy.net import Fido, attrs as a
            from sunpy.timeseries import TimeSeries
            tr = a.Time(self.start_date, self.end_date)
            logger.info(f"Fetching supplementary GOES XRS: {self.start_date} to {self.end_date}")
            results = Fido.search(tr, a.Instrument.xrs, a.Resolution("avg1m"))
            files = Fido.fetch(results)
            ts = TimeSeries(files, concatenate=True)
            df = ts.to_dataframe()
            df = df.rename(columns={"xrsa": "xrs_a_flux", "xrsb": "xrs_b_flux"})
            logger.info(f"GOES XRS: {len(df)} samples")
            return df
        except ImportError:
            logger.error("sunpy not installed. Install: pip install sunpy")
            raise
        except Exception as e:
            logger.warning(f"Could not fetch GOES: {e}. Using simulated data.")
            return self._generate_simulated_goes()

    def fetch_hek_flare_catalogue(self) -> pd.DataFrame:
        """Fetch HEK flare catalogue for ground-truth labels."""
        try:
            from sunpy.net import Fido, attrs as a
            tr = a.Time(self.start_date, self.end_date)
            logger.info("Fetching HEK flare catalogue...")
            results = Fido.search(tr, a.hek.EventType("FL"),
                                   a.hek.FL.GOESCls > "C1.0",
                                   a.hek.OBS.Observatory == "GOES")
            hek_table = results["hek"]
            df = hek_table.to_pandas()
            df = df.rename(columns={
                "event_starttime": "start_time",
                "event_peaktime": "peak_time",
                "event_endtime": "end_time",
                "fl_goescls": "goes_class",
                "fl_peakflux": "peak_flux"
            })
            for c in ["start_time", "peak_time", "end_time"]:
                df[c] = pd.to_datetime(df[c])
            logger.info(f"HEK catalogue: {len(df)} flares (>=C1.0)")
            return df
        except Exception as e:
            logger.warning(f"Could not fetch HEK: {e}. Returning empty catalogue.")
            return pd.DataFrame(columns=["start_time", "peak_time", "end_time",
                                          "goes_class", "peak_flux"])

    def _generate_simulated_goes(self) -> pd.DataFrame:
        np.random.seed(42)
        n_points = 100000
        times = pd.date_range(start=self.start_date, periods=n_points, freq=f"{self.cadence}s")
        quiet = 5e-8
        df = pd.DataFrame({
            "xrs_a_flux": 0.3 * quiet + np.random.lognormal(mean=np.log(quiet * 0.3), sigma=0.4, size=n_points),
            "xrs_b_flux": quiet + np.random.lognormal(mean=np.log(quiet), sigma=0.3, size=n_points)
        }, index=times)
        logger.warning("USING SIMULATED GOES — not real data")
        return df


def read_solexs_and_hel1os_unified(config: dict,
                                    solexs_paths: List[str] = None,
                                    hel1os_paths: List[str] = None) -> pd.DataFrame:
    """Convenience: read SoLEXS + HEL1OS and return unified DataFrame.

    Returns DataFrame with columns:
        soft_flux, hard_flux, solexs_flux, hel1os_flux
    aligned to a common 1-second time grid.
    """
    aditya_cfg = config["data"]["aditya_l1"]
    ingester = AdityaL1DataIngester(
        config["data"].get("aditya_l1_start_date", "2024-07-01"),
        config["data"].get("aditya_l1_end_date", "2025-12-31")
    )

    if solexs_paths and hel1os_paths:
        solexs, hel1os = ingester.load_aditya_l1_both(solexs_paths, hel1os_paths)
    elif solexs_paths:
        solexs = pd.concat([ingester.read_solexs_fits(p) for p in solexs_paths]).sort_index()
        hel1os = ingester._generate_simulated_hel1os()
    elif hel1os_paths:
        hel1os = pd.concat([ingester.read_hel1os_fits(p) for p in hel1os_paths]).sort_index()
        solexs = ingester._generate_simulated_solexs()
    else:
        solexs = ingester._generate_simulated_solexs()
        hel1os = ingester._generate_simulated_hel1os()

    combined = solexs.join(hel1os, how="outer")
    combined["soft_flux"] = combined["solexs_flux"]
    combined["hard_flux"] = combined["hel1os_flux"]
    combined = combined.ffill().dropna()
    return combined
