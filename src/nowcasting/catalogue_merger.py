"""
Flare catalogue merger — combines independent SoLEXS and HEL1OS detections
into one master catalogue.

This is an explicit deliverable in the problem statement:
    "Combine these independent catalogues to generate a master catalogue."

Merging logic:
    1. Detections within merge_window_seconds are treated as the same event
    2. Priority: use the detection with higher peak flux
    3. Tag each event with its source instrument(s)
"""

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FlareCatalogueMerger:
    """Merges independent soft and hard X-ray flare catalogues."""

    def __init__(self, merge_window_seconds: int = 60):
        self.merge_window = pd.Timedelta(seconds=merge_window_seconds)

    def merge(self, catalogues: List[pd.DataFrame],
              source_names: List[str]) -> pd.DataFrame:
        """Merge multiple instrument catalogues into one master catalogue.

        Args:
            catalogues: list of DataFrames from each instrument
            source_names: list of instrument names (e.g., ["SoLEXS", "HEL1OS"])
        Returns:
            master catalogue with deduplicated, merged events
        """
        if len(catalogues) == 0:
            return pd.DataFrame(columns=[
                "start_time", "peak_time", "end_time",
                "peak_flux", "goes_class", "sources"
            ])

        all_events = []
        for df, name in zip(catalogues, source_names):
            if len(df) > 0:
                df["source"] = name
                all_events.append(df)

        if len(all_events) == 0:
            return pd.DataFrame(columns=[
                "start_time", "peak_time", "end_time",
                "peak_flux", "goes_class", "sources"
            ])

        combined = pd.concat(all_events, ignore_index=True)
        combined = combined.sort_values("peak_time").reset_index(drop=True)

        master = []
        used = set()

        for i in range(len(combined)):
            if i in used:
                continue

            event = dict(combined.iloc[i])
            sources = [event["source"]]
            peak_flux = event["peak_flux"]
            peak_time = event["peak_time"]
            start_time = event["start_time"]
            end_time = event["end_time"]
            goes_class = event["goes_class"]

            for j in range(i + 1, len(combined)):
                if j in used:
                    continue
                other = combined.iloc[j]
                time_diff = abs((other["peak_time"] - peak_time).total_seconds())
                if time_diff <= self.merge_window.total_seconds():
                    used.add(j)
                    sources.append(other["source"])
                    if other["peak_flux"] > peak_flux:
                        peak_flux = other["peak_flux"]
                        peak_time = other["peak_time"]
                        goes_class = other["goes_class"]
                    start_time = min(start_time, other["start_time"])
                    end_time = max(end_time, other["end_time"])

            master.append({
                "start_time": start_time,
                "peak_time": peak_time,
                "end_time": end_time,
                "peak_flux": peak_flux,
                "goes_class": goes_class,
                "sources": "+".join(sorted(set(sources)))
            })

        master_df = pd.DataFrame(master)
        logger.info(
            f"Merged {sum(len(c) for c in catalogues)} detections -> "
            f"{len(master_df)} master events"
        )
        return master_df
