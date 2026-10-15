"""Attack traffic layered onto the benign baseline.

An attack is built as its own packet frame and then interleaved into a baseline, so the
output of one pass is valid input to the next. Baseline rows are never rewritten, which
keeps their benign labels correct as ground truth when several attacks stack on one
capture.
"""

from dataclasses import dataclass

import pandas as pd

from peruze.capture import schema, synthetic


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
    # Packets outside the capture window fall outside every rollup and score built from it, so the attack would be invisible rather than undetected.
    if start < first:
        raise ValueError(f"attack starts {start}, before the capture opens at {first}")
    end = start + pd.to_timedelta(duration_s, unit="s")
    if end > last:
        raise ValueError(f"attack ends {end}, after the capture closes at {last}")
    return start


@dataclass(frozen=True)
class PortScanConfig:
    """One external host probing many ports on a single internal target."""

    scanner: str = "198.51.100.66"
    # The target has to be a host the baseline already talks to, otherwise per host rollups show an address carrying attack traffic and nothing else, which is a far easier problem than the real one.
    target: str = synthetic.DEFAULT_HOSTS[0]
    # Inclusive bounds, matching the port list a scanner is handed.
    ports: tuple[int, int] = (1, 1024)
    port_count: int = 400
    offset_s: float = 120.0
    duration_s: float = 30.0

    def __post_init__(self) -> None:
        low, high = self.ports
        # One probe per port is what makes a scan a fan out, so a count the range cannot cover without repeats is a misconfiguration rather than something to clamp.
        if not 0 < self.port_count <= high - low + 1:
            raise ValueError(f"cannot probe {self.port_count} distinct ports in {low} to {high}")
