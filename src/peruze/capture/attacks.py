"""Attack traffic layered onto the benign baseline.

An attack is built as its own packet frame and then interleaved into a baseline, so the
output of one pass is valid input to the next. Baseline rows are never rewritten, which
keeps their benign labels correct as ground truth when several attacks stack on one
capture.
"""

import pandas as pd

from peruze.capture import schema


def interleave(baseline: pd.DataFrame, attack: pd.DataFrame) -> pd.DataFrame:
    """Place attack rows into a baseline by arrival time, under the packet contract."""
    combined = pd.concat([baseline, attack], ignore_index=True)
    # Timestamps can collide, and a stable sort puts the baseline packet first at those ties instead of leaving the row order undefined.
    return schema.validate_frame(combined.sort_values("ts", kind="stable", ignore_index=True))


def window_start(baseline: pd.DataFrame, offset_s: float, duration_s: float) -> pd.Timestamp:
    """The first arrival time of an attack, as an offset into the baseline's capture window."""
    if baseline.empty:
        raise ValueError("an empty baseline has no capture window to place an attack in")
    first, last = baseline["ts"].min(), baseline["ts"].max()
    start = first + pd.to_timedelta(offset_s, unit="s")
    end = start + pd.to_timedelta(duration_s, unit="s")
    # Packets outside the capture window fall outside every rollup and score built from it, so the attack would be invisible rather than undetected.
    if start < first:
        raise ValueError(f"attack starts {start}, before the capture opens at {first}")
    if end > last:
        raise ValueError(f"attack ends {end}, after the capture closes at {last}")
    return start
