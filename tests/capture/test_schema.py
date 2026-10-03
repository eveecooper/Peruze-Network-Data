"""The packet contract is what every capture source promises to return."""

import pandas as pd
import pytest

from peruze.capture import schema

TCP_RECORD = {
    "ts": "2026-01-01T12:00:00Z",
    "src_ip": "10.0.0.5",
    "dst_ip": "93.184.216.34",
    "src_port": 50512,
    "dst_port": 443,
    "protocol": "tcp",
    "length": 1460,
    "tcp_flags": "SA",
    "label": "benign",
}

ICMP_RECORD = {
    "ts": "2026-01-01T12:00:01Z",
    "src_ip": "10.0.0.5",
    "dst_ip": "93.184.216.34",
    "src_port": None,
    "dst_port": None,
    "protocol": "icmp",
    "length": 84,
    "tcp_flags": None,
    "label": None,
}


def dtypes_of(frame):
    return dict(frame.dtypes.astype(str))


def test_empty_frame_carries_the_declared_dtypes():
    frame = schema.empty_frame()

    assert frame.empty
    assert dtypes_of(frame) == schema.DTYPES


def test_frame_from_records_applies_the_declared_dtypes():
    frame = schema.frame_from_records([TCP_RECORD])

    assert list(frame.columns) == list(schema.DTYPES)
    assert dtypes_of(frame) == schema.DTYPES
    assert frame.loc[0, "ts"] == pd.Timestamp("2026-01-01T12:00:00Z")
    assert frame.loc[0, "dst_port"] == 443


def test_icmp_record_keeps_ports_and_flags_null():
    frame = schema.frame_from_records([ICMP_RECORD])

    assert frame["src_port"].isna().all()
    assert frame["dst_port"].isna().all()
    assert frame["tcp_flags"].isna().all()
    assert frame.loc[0, "length"] == 84


def test_frame_from_records_rejects_an_unknown_protocol():
    with pytest.raises(schema.SchemaError, match="protocol"):
        schema.frame_from_records([TCP_RECORD | {"protocol": "TCP"}])


def test_frame_from_records_rejects_an_unknown_label():
    with pytest.raises(schema.SchemaError, match="label"):
        schema.frame_from_records([TCP_RECORD | {"label": "portscan"}])


def test_validate_frame_restores_canonical_column_order():
    shuffled = schema.frame_from_records([TCP_RECORD])
    shuffled = shuffled[list(reversed(shuffled.columns))]

    assert list(schema.validate_frame(shuffled).columns) == list(schema.DTYPES)


def test_validate_frame_rejects_a_missing_column():
    frame = schema.frame_from_records([TCP_RECORD]).drop(columns=["tcp_flags"])

    with pytest.raises(schema.SchemaError, match="tcp_flags"):
        schema.validate_frame(frame)


def test_validate_frame_rejects_an_unexpected_column():
    frame = schema.frame_from_records([TCP_RECORD]).assign(ttl=64)

    with pytest.raises(schema.SchemaError, match="ttl"):
        schema.validate_frame(frame)


def test_validate_frame_rejects_a_wrong_dtype():
    frame = schema.frame_from_records([TCP_RECORD])
    frame["length"] = frame["length"].astype("float64")

    with pytest.raises(schema.SchemaError, match="length"):
        schema.validate_frame(frame)


def test_validate_frame_rejects_a_value_outside_the_vocabulary():
    frame = schema.frame_from_records([TCP_RECORD])
    frame["protocol"] = pd.Series(["sctp"]).astype("category")

    with pytest.raises(schema.SchemaError, match="sctp"):
        schema.validate_frame(frame)


def test_frame_survives_a_parquet_round_trip(tmp_path):
    frame = schema.frame_from_records([TCP_RECORD, ICMP_RECORD])
    path = tmp_path / "packets.parquet"
    frame.to_parquet(path, index=False)

    pd.testing.assert_frame_equal(schema.validate_frame(pd.read_parquet(path)), frame)


def test_frame_from_records_rejects_a_misspelled_field():
    with pytest.raises(schema.SchemaError, match="dest_port"):
        schema.frame_from_records([TCP_RECORD | {"dest_port": 443}])


def test_length_accepts_a_segment_above_65535():
    # Hosts with segmentation offload enabled report sizes far above one frame, so the column has to be wider than the IP total length field.
    frame = schema.frame_from_records([TCP_RECORD | {"length": 131072}])

    assert frame.loc[0, "length"] == 131072


def test_validate_frame_rejects_a_narrowed_vocabulary():
    frame = schema.frame_from_records([TCP_RECORD])
    frame["protocol"] = frame["protocol"].astype(pd.CategoricalDtype(["tcp"]))

    with pytest.raises(schema.SchemaError, match="categories"):
        schema.validate_frame(frame)
