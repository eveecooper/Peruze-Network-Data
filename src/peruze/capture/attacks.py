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
    # stable sort, so a baseline packet comes first at a colliding timestamp.
    return schema.validate_frame(combined.sort_values("ts", kind="stable", ignore_index=True))


def window_start(baseline: pd.DataFrame, offset_s: float, duration_s: float) -> pd.Timestamp:
    """The first arrival time of an attack, as an offset into the baseline's capture window."""
    if baseline.empty:
        raise ValueError("an empty baseline has no capture window to place an attack in")
    first, last = baseline["ts"].min(), baseline["ts"].max()
    start = first + pd.to_timedelta(offset_s, unit="s")
    # outside the window the attack misses every rollup, so it reads as invisible, not undetected.
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
    # a host the baseline already talks to, or per host rollups show an address with nothing but attack traffic.
    target: str = synthetic.DEFAULT_HOSTS[0]
    # Inclusive bounds, matching the port list a scanner is handed.
    ports: tuple[int, int] = (1, 1024)
    port_count: int = 400
    # one port for the whole sweep, so probes differ only in destination.
    source_port: int = 50123
    offset_s: float = 120.0
    duration_s: float = 30.0

    def __post_init__(self) -> None:
        low, high = self.ports
        # bounds before count: port 0 is reserved, and anything past the field width fails later as a uint16 cast.
        if not 1 <= low <= high <= 65535:
            raise ValueError(f"ports {low} to {high} must be an ascending range inside 1 to 65535")
        # one probe per port, so a count the range cannot cover is a misconfiguration, not something to clamp.
        if not 0 < self.port_count <= high - low + 1:
            raise ValueError(f"cannot probe {self.port_count} distinct ports in {low} to {high}")
        # scanners bind an ephemeral port. 49152 is the low bound of the range.
        if not 49152 <= self.source_port <= 65535:
            raise ValueError(f"source port {self.source_port} is not ephemeral, 49152 to 65535")


def probe_ports(config: PortScanConfig) -> np.ndarray:
    """The destination ports a scan touches, one probe each."""
    low, _ = config.ports
    # sweep from the low bound, where the well known services sit, so the block is contiguous.
    return np.arange(low, low + config.port_count)


def probe_times(config: PortScanConfig, start: pd.Timestamp) -> pd.DatetimeIndex:
    """Probe arrival times across the scan window."""
    # drop the right edge: gaps stay duration / port_count and the last probe stays inside the window.
    offsets = np.linspace(0.0, config.duration_s, config.port_count, endpoint=False)
    return start + pd.to_timedelta(offsets, unit="s")


def probe_frame(config: PortScanConfig, start: pd.Timestamp) -> pd.DataFrame:
    """A bare SYN per probed port, ready to interleave into a baseline."""
    return schema.frame_from_columns(
        {
            "ts": probe_times(config, start),
            "src_ip": config.scanner,
            "dst_ip": config.target,
            "src_port": config.source_port,
            "dst_port": probe_ports(config),
            "protocol": "tcp",
            # 54 bytes: ip and tcp headers with no options and no payload, same for every probe.
            "length": 54,
            "tcp_flags": "S",
            "label": "port_scan",
        }
    )


@dataclass(frozen=True)
class DDoSConfig:
    """Many external hosts flooding one target port over a short, high volume burst."""

    target: str = synthetic.DEFAULT_HOSTS[0]
    target_port: int = 443
    # a volumetric flood, not a handshake attempt, so tcp or udp covers the realistic cases.
    protocol: str = "udp"
    host_count: int = 200
    packet_count: int = 20000
    offset_s: float = 150.0
    duration_s: float = 10.0

    def __post_init__(self) -> None:
        if self.protocol not in ("tcp", "udp"):
            raise ValueError(f"protocol {self.protocol!r} must be tcp or udp")
        if not 1 <= self.target_port <= 65535:
            raise ValueError(f"target port {self.target_port} must be inside 1 to 65535")
        if self.host_count < 1:
            raise ValueError(f"host_count {self.host_count} must be at least 1")
        # a botnet host that never sends a packet is not part of the flood.
        if self.packet_count < self.host_count:
            raise ValueError(
                f"cannot spread {self.packet_count} packets across {self.host_count} hosts, "
                "one packet each minimum"
            )


def flood_sources(config: DDoSConfig) -> np.ndarray:
    """One attacking address per host, standing in for a botnet."""
    # TEST-NET-3 per RFC 5737, 254 usable addresses. Past that the pool repeats, which
    # matches a NAT'd botnet reusing a public address for several infected hosts.
    offsets = np.arange(config.host_count) % 254 + 1
    return np.array([f"203.0.113.{offset}" for offset in offsets])


def flood_times(config: DDoSConfig, start: pd.Timestamp) -> pd.DatetimeIndex:
    """Packet arrival times across the flood window."""
    # drop the right edge: gaps stay duration / packet_count and the last packet stays inside the window.
    offsets = np.linspace(0.0, config.duration_s, config.packet_count, endpoint=False)
    return start + pd.to_timedelta(offsets, unit="s")


def flood_frame(config: DDoSConfig, start: pd.Timestamp) -> pd.DataFrame:
    """One packet per flood arrival, source host cycling round robin across the botnet."""
    sources = flood_sources(config)
    hosts = np.arange(config.packet_count) % config.host_count
    tcp = config.protocol == "tcp"
    return schema.frame_from_columns(
        {
            "ts": flood_times(config, start),
            "src_ip": sources[hosts],
            "dst_ip": config.target,
            # each host keeps one ephemeral port for the burst, same as a single real connection would.
            "src_port": 49152 + hosts % (65536 - 49152),
            "dst_port": config.target_port,
            "protocol": config.protocol,
            # 54 bytes: a bare SYN, same header size as the port scan probes. 28 bytes: ip plus udp headers, no payload.
            "length": 54 if tcp else 28,
            "tcp_flags": "S" if tcp else pd.NA,
            "label": "ddos",
        }
    )


def generate_port_scan(
    baseline: pd.DataFrame, config: PortScanConfig | None = None
) -> pd.DataFrame:
    """Layer one port scan onto a baseline capture, at the offset the config names."""
    config = config or PortScanConfig()
    start = window_start(baseline, config.offset_s, config.duration_s)
    return interleave(baseline, probe_frame(config, start))


@dataclass(frozen=True)
class BeaconConfig:
    """One internal host checking in with an external command and control address on a fixed interval."""

    source: str = synthetic.DEFAULT_HOSTS[0]
    # TEST-NET-1 per RFC 5737, kept distinct from the scanner and botnet ranges elsewhere in this module.
    c2: str = "192.0.2.10"
    c2_port: int = 443
    interval_s: float = 60.0
    # random spread around the interval, so checkins do not land on an exact, suspiciously regular grid.
    jitter_s: float = 5.0
    beacon_count: int = 20
    offset_s: float = 60.0

    def __post_init__(self) -> None:
        if not 1 <= self.c2_port <= 65535:
            raise ValueError(f"c2 port {self.c2_port} must be inside 1 to 65535")
        if self.interval_s <= 0:
            raise ValueError(f"interval_s {self.interval_s} must be positive")
        # jitter at or past the interval could push a checkin ahead of the one before it.
        if not 0 <= self.jitter_s < self.interval_s:
            raise ValueError(
                f"jitter_s {self.jitter_s} must be smaller than interval_s {self.interval_s}"
            )
        if self.beacon_count < 1:
            raise ValueError(f"beacon_count {self.beacon_count} must be at least 1")


def generate_ddos_burst(baseline: pd.DataFrame, config: DDoSConfig | None = None) -> pd.DataFrame:
    """Layer one DDoS burst onto a baseline capture, at the offset the config names."""
    config = config or DDoSConfig()
    start = window_start(baseline, config.offset_s, config.duration_s)
    return interleave(baseline, flood_frame(config, start))
