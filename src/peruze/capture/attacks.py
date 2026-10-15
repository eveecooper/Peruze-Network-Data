"""Attack traffic layered onto the benign baseline.

An attack is built as its own packet frame and then interleaved into a baseline, so the
output of one pass is valid input to the next. Baseline rows are never rewritten, which
keeps their benign labels correct as ground truth when several attacks stack on one
capture.
"""

from dataclasses import dataclass

import numpy as np
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
        # A scanner is handed a real port range, so the bounds are checked before the count: port 0 is reserved, and a port wider than the field would surface as a failed UInt16 cast rather than a message naming the field.
        if not 1 <= low <= high <= 65535:
            raise ValueError(f"ports {low} to {high} must be an ascending range inside 1 to 65535")
        # One probe per port is what makes a scan a fan out, so a count the range cannot cover without repeats is a misconfiguration rather than something to clamp.
        if not 0 < self.port_count <= high - low + 1:
            raise ValueError(f"cannot probe {self.port_count} distinct ports in {low} to {high}")


def probe_ports(config: PortScanConfig) -> np.ndarray:
    """The destination ports a scan touches, one probe each."""
    low, _ = config.ports
    # A naive sweep walks the range from its low bound, where the well known services sit, so the probed block is contiguous rather than scattered.
    return np.arange(low, low + config.port_count)


def probe_times(config: PortScanConfig, start: pd.Timestamp) -> pd.DatetimeIndex:
    """Probe arrival times across the scan window."""
    # drop the right edge: gaps stay duration / port_count and the last probe stays inside the window.
    offsets = np.linspace(0.0, config.duration_s, config.port_count, endpoint=False)
    return start + pd.to_timedelta(offsets, unit="s")
