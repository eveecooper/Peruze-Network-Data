"""Synthetic baseline traffic.

Depending on a real capture is awkward: the files are large, they carry private
addresses, and no two of them look alike. The generator hands every later section a
labeled frame that is identical on every run for a given seed, so a detector's score
can be compared across commits and a failing test means a real regression rather than
a different sample. Attack traffic gets layered onto this baseline by the modules that
follow, which is why nothing here produces anything but benign rows.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from peruze.capture import schema

TCP_FLAGS = ("A", "PA", "S", "SA", "FA")
TCP_FLAG_WEIGHTS = (0.55, 0.30, 0.06, 0.06, 0.03)

# Handshake and teardown packets carry no payload, so these flags cap a packet's size.
CONTROL_FLAGS = ("S", "SA", "FA")


@dataclass(frozen=True)
class Service:
    """An endpoint the internal hosts talk to, and how much of the traffic it carries."""

    host: str
    port: int | None
    protocol: str
    weight: float

    def __post_init__(self) -> None:
        if self.protocol not in schema.PROTOCOLS:
            raise ValueError(f"protocol {self.protocol!r} is not one of {schema.PROTOCOLS}")
        if self.weight <= 0:
            raise ValueError(f"weight must be positive, got {self.weight}")


DEFAULT_HOSTS = ("10.0.0.11", "10.0.0.12", "10.0.0.23", "10.0.0.37")

# RFC 5737 reserves 192.0.2.0/24, 198.51.100.0/24, and 203.0.113.0/24 for documentation, so using them keeps the repo from naming a host that belongs to somebody.
DEFAULT_SERVICES = (
    Service("203.0.113.10", 443, "tcp", 6.0),
    Service("203.0.113.47", 443, "tcp", 3.0),
    Service("10.0.0.1", 53, "udp", 2.0),
    Service("192.0.2.9", 22, "tcp", 0.5),
    Service("198.51.100.14", None, "icmp", 0.2),
)


@dataclass(frozen=True)
class BaselineConfig:
    """Shape of the benign traffic to generate."""

    start: pd.Timestamp = pd.Timestamp("2026-01-01T00:00:00Z")
    duration_s: float = 600.0
    packets_per_second: float = 20.0
    data_packet_share: float = 0.35
    hosts: tuple[str, ...] = DEFAULT_HOSTS
    services: tuple[Service, ...] = DEFAULT_SERVICES
    seed: int = 0


def generate_baseline(config: BaselineConfig | None = None) -> pd.DataFrame:
    """Build a benign packet frame, identical on every run for a given seed."""
    config = config or BaselineConfig()
    rng = np.random.default_rng(config.seed)

    ts = _arrival_times(rng, config)
    count = len(ts)
    chosen = _choose_services(rng, config, count)

    protocol = np.array([config.services[index].protocol for index in chosen])
    server_host = np.array([config.services[index].host for index in chosen])
    server_port = np.array([config.services[index].port for index in chosen], dtype=object)

    # Both directions of every conversation appear, because the flow aggregation in section 02 has to pair them up and the throughput charts would be half empty otherwise.
    outbound = rng.random(count) < 0.5
    client_host = rng.choice(config.hosts, size=count)
    client_port = rng.integers(49152, 65536, size=count)

    is_tcp = protocol == "tcp"
    carries_ports = protocol != "icmp"
    flags = rng.choice(TCP_FLAGS, size=count, p=TCP_FLAG_WEIGHTS)

    return schema.frame_from_columns(
        {
            "ts": ts,
            "src_ip": np.where(outbound, client_host, server_host),
            "dst_ip": np.where(outbound, server_host, client_host),
            "src_port": _nullable_ports(np.where(outbound, client_port, server_port), carries_ports),
            "dst_port": _nullable_ports(np.where(outbound, server_port, client_port), carries_ports),
            "protocol": protocol,
            "length": _packet_sizes(rng, config, is_tcp & np.isin(flags, CONTROL_FLAGS)),
            "tcp_flags": _tcp_flag_column(flags, is_tcp),
            "label": np.full(count, "benign"),
        }
    )


def _arrival_times(rng: np.random.Generator, config: BaselineConfig) -> pd.DatetimeIndex:
    """Packet arrival times with exponential gaps, which is what a Poisson process gives.

    Evenly spaced packets would hand the rolling detectors in section 04 a baseline with
    no variance at all, and then any burst whatsoever would clear any threshold. The
    jitter here is what makes choosing a threshold a real decision instead of a formality.

    Gaps are oversampled and then cut at the window edge, because the number of arrivals
    in a fixed window is itself random, and drawing exactly the expected count would
    leave the tail of the window short.
    """
    expected = config.duration_s * config.packets_per_second
    gaps = rng.exponential(1.0 / config.packets_per_second, size=int(expected * 1.25) + 64)
    offsets = np.cumsum(gaps)
    return config.start + pd.to_timedelta(offsets[offsets < config.duration_s], unit="s")


def _choose_services(rng: np.random.Generator, config: BaselineConfig, count: int) -> np.ndarray:
    """Pick a service per packet, in proportion to the configured weights."""
    weights = np.array([service.weight for service in config.services], dtype=float)
    return rng.choice(len(config.services), size=count, p=weights / weights.sum())


def _packet_sizes(
    rng: np.random.Generator, config: BaselineConfig, payload_free: np.ndarray
) -> np.ndarray:
    """Sizes drawn from two modes rather than one spread.

    Real traffic is bimodal: small acknowledgements and handshakes on one side, close to
    full MTU payloads on the other, with very little in between. A single normal
    distribution would make the autoencoder in section 07 look far better than it
    deserves, because it would only ever have to learn one hump.

    Rows flagged payload_free never draw from the data mode. A twelve hundred byte SYN
    would be nonsense to anyone reading the output, and it would hand section 06 a
    feature combination that cannot occur on a real wire.
    """
    count = len(payload_free)
    data = (rng.random(count) < config.data_packet_share) & ~payload_free
    return np.where(
        data,
        rng.integers(1100, 1515, size=count),
        rng.integers(54, 121, size=count),
    )


def _tcp_flag_column(
    flags: np.ndarray, is_tcp: np.ndarray
) -> pd.api.extensions.ExtensionArray:
    """Flag letters on the TCP rows, blank everywhere else."""
    column = pd.array(flags, dtype="string")
    column[~is_tcp] = pd.NA
    return column


def _nullable_ports(
    values: np.ndarray, carries_ports: np.ndarray
) -> pd.api.extensions.ExtensionArray:
    """Ports as a nullable column, blank wherever the protocol has no port to report."""
    ports = pd.array(values, dtype="UInt16")
    ports[~carries_ports] = pd.NA
    return ports
