"""The baseline generator has to be reproducible and has to satisfy the packet contract."""

import pytest

from peruze.capture import schema, synthetic

# A minute is long enough for every property below and keeps the suite quick.
FAST = synthetic.BaselineConfig(duration_s=60.0)


def test_baseline_satisfies_the_packet_contract():
    frame = synthetic.generate_baseline(FAST)

    assert not frame.empty
    assert dict(frame.dtypes.astype(str)) == schema.DTYPES
    schema.validate_frame(frame)


def test_the_same_seed_gives_the_same_frame():
    assert synthetic.generate_baseline(FAST).equals(synthetic.generate_baseline(FAST))


def test_a_different_seed_gives_a_different_frame():
    other = synthetic.BaselineConfig(duration_s=60.0, seed=FAST.seed + 1)

    assert not synthetic.generate_baseline(FAST).equals(synthetic.generate_baseline(other))


def test_every_row_is_labelled_benign():
    frame = synthetic.generate_baseline(FAST)

    assert (frame["label"] == "benign").all()


def test_timestamps_are_ordered_and_inside_the_window():
    frame = synthetic.generate_baseline(FAST)
    elapsed = (frame["ts"] - FAST.start).dt.total_seconds()

    assert frame["ts"].is_monotonic_increasing
    assert elapsed.min() >= 0
    assert elapsed.max() < FAST.duration_s


def test_the_packet_rate_is_near_the_configured_rate():
    frame = synthetic.generate_baseline(FAST)
    rate = len(frame) / FAST.duration_s

    assert rate == pytest.approx(FAST.packets_per_second, rel=0.1)


def test_icmp_rows_carry_no_ports_and_no_flags():
    frame = synthetic.generate_baseline(FAST)
    icmp = frame[frame["protocol"] == "icmp"]

    assert not icmp.empty
    assert icmp[["src_port", "dst_port", "tcp_flags"]].isna().all().all()


def test_flags_appear_on_tcp_rows_only():
    frame = synthetic.generate_baseline(FAST)

    assert frame.loc[frame["protocol"] == "tcp", "tcp_flags"].notna().all()
    assert frame.loc[frame["protocol"] != "tcp", "tcp_flags"].isna().all()


def test_packet_sizes_are_bimodal():
    # Nothing lands in the middle of the range, which is what later sections assume.
    length = synthetic.generate_baseline(FAST)["length"]

    assert (length < 200).any()
    assert (length > 1000).any()
    assert not ((length >= 200) & (length <= 1000)).any()


def test_handshake_packets_carry_no_payload():
    frame = synthetic.generate_baseline(FAST)
    control = frame[frame["tcp_flags"].isin(synthetic.CONTROL_FLAGS)]

    assert not control.empty
    assert control["length"].max() < 200


def test_only_the_configured_endpoints_appear():
    frame = synthetic.generate_baseline(FAST)
    allowed = set(FAST.hosts) | {service.host for service in FAST.services}

    assert set(frame["src_ip"]) <= allowed
    assert set(frame["dst_ip"]) <= allowed


def test_service_rejects_a_protocol_outside_the_contract():
    with pytest.raises(ValueError, match="sctp"):
        synthetic.Service("203.0.113.10", 443, "sctp", 1.0)


def test_service_rejects_a_non_positive_weight():
    with pytest.raises(ValueError, match="weight"):
        synthetic.Service("203.0.113.10", 443, "tcp", 0.0)
