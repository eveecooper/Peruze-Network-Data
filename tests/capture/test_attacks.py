"""Interleaving attack rows must leave the baseline intact and keep the frame in time order."""

import pandas as pd

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
