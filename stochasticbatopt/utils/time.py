from __future__ import annotations
import pandas as pd


def build_time_index(
    start : str | pd.Timestamp,
    end   : str | pd.Timestamp,
    freq  : str = "h",
) -> pd.DatetimeIndex:
    """Hourly UTC DatetimeIndex on [start, end) — left-inclusive."""
    return pd.date_range(
        start=pd.Timestamp(start, tz="UTC"),
        end  =pd.Timestamp(end,   tz="UTC"),
        freq =freq, inclusive="left",
    )
