"""
Real-time nowcasting (detection) module using threshold + derivative triggers.

Implements the standard operational approach:
    1. Flux exceeds k*background_level
    2. dF/dt exceeds threshold for n consecutive samples
    3. Classify flare class from peak flux (GOES-equivalent scale)
    4. Cooldown to prevent duplicate triggers

This matches NOAA SWPC's operational method and the problem statement's
requirement for a "threshold + derivative trigger" nowcasting module.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ThresholdFlareDetector:
    """Threshold-based flare detector for a single X-ray channel.

    Detects flares when:
        1. Flux > background_level * threshold_sigma for n consecutive samples
        2. dF/dt exceeds dFdt_threshold_sigma for n consecutive samples

    GOES flare class boundaries (W/m^2 in 1-8 A band):
        A: < 1e-7    B: [1e-7, 1e-6)
        C: [1e-6, 1e-5)    M: [1e-5, 1e-4)    X: >= 1e-4
    """

    GOES_CLASS_BOUNDARIES = {
        "A": (0, 1e-7),
        "B": (1e-7, 1e-6),
        "C": (1e-6, 1e-5),
        "M": (1e-5, 1e-4),
        "X": (1e-4, np.inf)
    }

    def __init__(self, channel_name: str, config: dict):
        self.channel_name = channel_name
        self.cfg = config["nowcasting"]
        self.flux_sigma = self.cfg["soft_channel_threshold_sigma"]
        self.dfdt_sigma = self.cfg["dFdt_threshold_sigma"]
        self.n_consecutive = self.cfg["consecutive_samples"]
        self.cooldown_min = self.cfg["cooldown_minutes"]

        self.background_level = None
        self.flux_noise_sigma = None
        self.dfdt_noise_sigma = None
        self._last_detection = None

    def estimate_noise(self, flux: np.ndarray):
        """Estimate background level and noise from quiet periods.

        Uses the lowest 50th percentile of flux as background estimate
        (similar to trailing percentile approach in preprocessing).
        """
        sorted_flux = np.sort(flux)
        quiet_idx = int(len(sorted_flux) * 0.3)
        quiet_samples = sorted_flux[:max(quiet_idx, 100)]

        self.background_level = np.median(quiet_samples)
        self.flux_noise_sigma = np.std(quiet_samples) + 1e-12

        dfdt = np.gradient(flux)
        quiet_dfdt = dfdt[:max(quiet_idx, 100)]
        self.dfdt_noise_sigma = np.std(quiet_dfdt) + 1e-12

        logger.info(
            f"[{self.channel_name}] Background={self.background_level:.2e}, "
            f"noise_sigma={self.flux_noise_sigma:.2e}"
        )

    def detect(self, times: pd.DatetimeIndex,
               flux: np.ndarray) -> pd.DataFrame:
        """Run threshold-based flare detection.

        Returns DataFrame with columns:
            start_time, peak_time, end_time, peak_flux, goes_class, channel
        """
        if self.background_level is None:
            self.estimate_noise(flux)

        flux_threshold = self.background_level + self.flux_sigma * self.flux_noise_sigma
        dfdt = np.gradient(flux)
        dfdt_threshold = self.dfdt_sigma * self.dfdt_noise_sigma

        above_flux = flux > flux_threshold
        above_dfdt = np.abs(dfdt) > dfdt_threshold
        triggered = above_flux & above_dfdt

        detections = self._parse_triggers(times, triggered, flux)
        detections = self._apply_cooldown(detections)
        detections["channel"] = self.channel_name

        logger.info(
            f"[{self.channel_name}] Detected {len(detections)} flare events"
        )
        return detections

    def _parse_triggers(self, times: pd.DatetimeIndex,
                        triggered: np.ndarray, flux: np.ndarray) -> pd.DataFrame:
        """Parse boolean trigger array into discrete flare events."""
        events = []
        in_flare = False
        consecutive_count = 0
        start_idx = None
        peak_flux = 0.0
        peak_idx = None

        for i in range(len(triggered)):
            if triggered[i]:
                if not in_flare:
                    consecutive_count += 1
                    if consecutive_count >= self.n_consecutive:
                        in_flare = True
                        start_idx = i - self.n_consecutive + 1
                        peak_flux = flux[i]
                        peak_idx = i
                else:
                    if flux[i] > peak_flux:
                        peak_flux = flux[i]
                        peak_idx = i
            else:
                if in_flare:
                    if flux[i] < self.background_level * 2:
                        events.append({
                            "start_time": times[max(0, start_idx)],
                            "peak_time": times[peak_idx],
                            "end_time": times[i],
                            "peak_flux": peak_flux,
                            "goes_class": self._classify_flare(peak_flux)
                        })
                        in_flare = False
                        consecutive_count = 0
                        start_idx = None
                        peak_flux = 0.0
                        peak_idx = None
                else:
                    consecutive_count = 0

        if in_flare:
            events.append({
                "start_time": times[max(0, start_idx)],
                "peak_time": times[peak_idx],
                "end_time": times[-1],
                "peak_flux": peak_flux,
                "goes_class": self._classify_flare(peak_flux)
            })

        return pd.DataFrame(events) if events else pd.DataFrame(
            columns=["start_time", "peak_time", "end_time",
                     "peak_flux", "goes_class"]
        )

    def _apply_cooldown(self, detections: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate detections within cooldown window."""
        if len(detections) < 2:
            return detections
        cooldown = pd.Timedelta(minutes=self.cooldown_min)
        mask = [True]
        last_peak = detections.iloc[0]["peak_time"]
        for i in range(1, len(detections)):
            if (detections.iloc[i]["peak_time"] - last_peak) >= cooldown:
                mask.append(True)
                last_peak = detections.iloc[i]["peak_time"]
            else:
                mask.append(False)
        return detections[mask].reset_index(drop=True)

    def _classify_flare(self, peak_flux: float) -> str:
        for cls, (lo, hi) in self.GOES_CLASS_BOUNDARIES.items():
            if lo <= peak_flux < hi:
                return cls
        return "A"


class Nowcaster:
    """Combined nowcaster using both SoLEXS (soft) and HEL1OS (hard) channels."""

    def __init__(self, config: dict):
        self.cfg = config["nowcasting"]
        self.soft_detector = ThresholdFlareDetector("SoLEXS", config)
        self.hard_detector = ThresholdFlareDetector("HEL1OS", config)

    def nowcast(self, times: pd.DatetimeIndex, soft_flux: np.ndarray,
                hard_flux: np.ndarray) -> pd.DataFrame:
        """Run nowcasting on both channels independently.

        Returns merged catalogue with both soft and hard detections.
        """
        soft_events = self.soft_detector.detect(times, soft_flux)
        hard_events = self.hard_detector.detect(times, hard_flux)

        combined = pd.concat([soft_events, hard_events], ignore_index=True)
        combined = combined.sort_values("peak_time").reset_index(drop=True)

        logger.info(
            f"Nowcaster: {len(soft_events)} SoLEXS + {len(hard_events)} "
            f"HEL1OS = {len(combined)} total events"
        )
        return combined
