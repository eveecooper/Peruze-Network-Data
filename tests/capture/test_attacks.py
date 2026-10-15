"""Interleaving attack rows must leave the baseline intact and keep the frame in time order."""

import pandas as pd
import pytest

from peruze.capture import attacks, schema, synthetic

BASELINE = synthetic.generate_baseline(synthetic.BaselineConfig(duration_s=60.0))
START = BASELINE["ts"].iloc[0]


def _attack_rows(*timestamps: pd.Timestamp) -> pd.DataFrame:
    """One tiny SYN per timestamp."""
    return schema.frame_from_records(
        {
            "ts": timestamp,
            "src_ip": "198.51.100.66",
            "dst_ip": "10.0.0.11",
            "src_port": 55610,
            "dst_port": 22,
            "protocol": "tcp",
            "length": 58,
            "tcp_flags": "S",
            "label": "port_scan",
        }
        for timestamp in timestamps
    )


def test_the_combined_frame_satisfies_the_packet_contract():
    frame = attacks.interleave(BASELINE, _attack_rows(START + pd.Timedelta(seconds=10)))

    assert dict(frame.dtypes.astype(str)) == schema.DTYPES
    schema.validate_frame(frame)


def test_the_combined_frame_comes_back_in_time_order():
    unordered = (41.0, 7.0, 23.0)
    frame = attacks.interleave(
        BASELINE, _attack_rows(*(START + pd.Timedelta(seconds=at) for at in unordered))
    )

    assert frame["ts"].is_monotonic_increasing


def test_the_baseline_rows_come_through_untouched():
    frame = attacks.interleave(BASELINE, _attack_rows(START + pd.Timedelta(seconds=10)))
    benign = frame[frame["label"] == "benign"].reset_index(drop=True)

    assert benign.equals(BASELINE)


def test_every_attack_row_survives():
    attack = _attack_rows(*(START + pd.Timedelta(seconds=at) for at in (10.0, 20.0, 30.0)))
    frame = attacks.interleave(BASELINE, attack)

    assert len(frame) == len(BASELINE) + len(attack)


def test_a_baseline_packet_sorts_first_at_a_colliding_timestamp():
    collision = BASELINE["ts"].iloc[5]
    frame = attacks.interleave(BASELINE, _attack_rows(collision))

    assert list(frame.loc[frame["ts"] == collision, "label"]) == ["benign", "port_scan"]


def test_the_window_starts_at_the_offset_into_the_capture():
    assert attacks.window_start(BASELINE, 10.0, 5.0) == START + pd.Timedelta(seconds=10)


def test_a_zero_offset_starts_on_the_first_packet():
    assert attacks.window_start(BASELINE, 0.0, 5.0) == START


def test_a_window_filling_the_rest_of_the_capture_is_accepted():
    span = (BASELINE["ts"].max() - START).total_seconds()

    assert attacks.window_start(BASELINE, 0.0, span) == START


def test_a_window_running_past_the_last_packet_is_rejected():
    span = (BASELINE["ts"].max() - START).total_seconds()

    with pytest.raises(ValueError, match="after the capture"):
        attacks.window_start(BASELINE, 0.0, span + 1.0)


def test_a_window_starting_before_the_first_packet_is_rejected():
    with pytest.raises(ValueError, match="before the capture"):
        attacks.window_start(BASELINE, -1.0, 5.0)


def test_an_empty_baseline_has_no_window_to_place():
    with pytest.raises(ValueError, match="empty"):
        attacks.window_start(schema.empty_frame(), 10.0, 5.0)


def test_probing_more_ports_than_the_range_holds_is_rejected():
    with pytest.raises(ValueError, match="distinct ports"):
        attacks.PortScanConfig(ports=(80, 90), port_count=50)


def test_a_scan_probes_the_configured_number_of_ports():
    assert len(attacks.probe_ports(attacks.PortScanConfig(port_count=12))) == 12


def test_every_probed_port_is_distinct():
    ports = attacks.probe_ports(attacks.PortScanConfig())

    assert len(set(ports.tolist())) == len(ports)


def test_a_full_range_sweep_covers_both_bounds():
    ports = attacks.probe_ports(attacks.PortScanConfig(ports=(20, 120), port_count=101))

    assert (ports.min(), ports.max()) == (20, 120)
