"""The packet record contract.

The synthetic generator, the pcap reader, and the live sniffer all return the
same columns with the same dtypes, so no stage after capture has to guess what
it was handed. Sources build plain row dicts and call frame_from_records.
validate_frame is the gate for a frame that arrived from somewhere else, such
as a parquet file written by an earlier run.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

PROTOCOLS = ("tcp", "udp", "icmp", "other")

# Ground truth for the supervised phase. Only the generator knows which attack a packet belongs to, so captures off the wire leave the label null.
LABELS = ("benign", "port_scan", "ddos", "beacon", "slow_exfil")


class SchemaError(ValueError):
    """A record or a frame does not satisfy the packet contract."""


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str
    note: str


COLUMNS: tuple[Column, ...] = (
    Column("ts", "datetime64[ns, UTC]", "arrival time, tz aware so resampling cannot drift"),
    Column("src_ip", "string", "source address"),
    Column("dst_ip", "string", "destination address"),
    Column("src_port", "UInt16", "nullable, ICMP carries no ports"),
    Column("dst_port", "UInt16", "nullable, and the scan detectors key on it"),
    Column("protocol", "category", "one of PROTOCOLS"),
    Column("length", "UInt32", "bytes on the wire, 32 bit like the pcap field it comes from"),
    Column("tcp_flags", "string", "flag letters such as S or SA, null off TCP"),
    Column("label", "category", "ground truth from LABELS, null on real captures"),
)

DTYPES: dict[str, str] = {column.name: column.dtype for column in COLUMNS}

# Both category columns are closed vocabularies. Pandas turns an unlisted value into a null instead of complaining, which would hide a parser typo for several phases, so the values get checked before the cast rather than after.
VOCABULARIES: dict[str, tuple[str, ...]] = {"protocol": PROTOCOLS, "label": LABELS}


def empty_frame() -> pd.DataFrame:
    """A zero row frame that still carries the contract's dtypes."""
    return frame_from_records([])


def frame_from_records(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Build a valid packet frame from row dicts.

    A record may omit a column, which lands as null, but it may not invent one:
    a misspelled key would otherwise become a silently empty column.
    """
    frame = pd.DataFrame(list(records))
    _reject_unexpected(frame.columns)
    frame = frame.reindex(columns=list(DTYPES))
    for name, vocabulary in VOCABULARIES.items():
        _reject_unlisted(frame[name], name, vocabulary)
    for name in DTYPES:
        frame[name] = frame[name].astype(_cast_target(name))
    return frame


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the frame in canonical column order, or raise SchemaError."""
    missing = [name for name in DTYPES if name not in frame.columns]
    if missing:
        raise SchemaError(f"missing columns: {', '.join(missing)}")
    _reject_unexpected(frame.columns)
    wrong = [
        complaint
        for name in DTYPES
        if (complaint := _dtype_complaint(frame[name], name)) is not None
    ]
    if wrong:
        raise SchemaError(f"wrong dtypes: {'; '.join(wrong)}")
    return frame[list(DTYPES)]


def _dtype_complaint(values: pd.Series, name: str) -> str | None:
    """Describe how a column departs from the contract, or None if it conforms."""
    dtype = values.dtype
    if str(dtype) != DTYPES[name]:
        return f"{name} is {dtype}, expected {DTYPES[name]}"
    vocabulary = VOCABULARIES.get(name)
    # "category" says nothing about which categories a column actually holds. A frame carrying a narrowed vocabulary would quietly drop groups from every later groupby rather than reporting them as zero, so compare the values.
    if vocabulary is not None and tuple(dtype.categories) != vocabulary:
        return f"{name} categories are {list(dtype.categories)}, expected {list(vocabulary)}"
    return None


def _cast_target(name: str) -> Any:
    vocabulary = VOCABULARIES.get(name)
    if vocabulary is None:
        return DTYPES[name]
    return pd.CategoricalDtype(vocabulary)


def _reject_unexpected(names: Iterable[Any]) -> None:
    unexpected = sorted(str(name) for name in names if name not in DTYPES)
    if unexpected:
        raise SchemaError(f"unexpected columns: {', '.join(unexpected)}")


def _reject_unlisted(values: pd.Series, name: str, vocabulary: tuple[str, ...]) -> None:
    unlisted = sorted(str(value) for value in set(values.dropna().unique()) - set(vocabulary))
    if unlisted:
        raise SchemaError(f"{name} values outside its vocabulary: {', '.join(unlisted)}")
