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


def test_a_port_range_reaching_past_the_field_width_is_rejected():
    with pytest.raises(ValueError, match="ascending range"):
        attacks.PortScanConfig(ports=(65000, 70000), port_count=10)


def test_a_port_range_starting_at_the_reserved_port_is_rejected():
    with pytest.raises(ValueError, match="ascending range"):
        attacks.PortScanConfig(ports=(0, 100), port_count=10)


def test_a_descending_port_range_is_rejected():
    with pytest.raises(ValueError, match="ascending range"):
        attacks.PortScanConfig(ports=(900, 80), port_count=10)


def test_a_scan_sends_one_probe_per_port():
    config = attacks.PortScanConfig(port_count=10, duration_s=5.0)

    assert len(attacks.probe_times(config, START)) == 10


def test_the_probes_run_on_an_even_grid_from_the_window_start():
    times = attacks.probe_times(attacks.PortScanConfig(port_count=4, duration_s=8.0), START)

    assert times[0] == START
    assert list(times[1:] - times[:-1]) == [pd.Timedelta(seconds=2.0)] * 3


def test_the_last_probe_lands_before_the_window_closes():
    times = attacks.probe_times(attacks.PortScanConfig(port_count=10, duration_s=5.0), START)

    assert times.max() < START + pd.Timedelta(seconds=5.0)


def test_a_source_port_below_the_ephemeral_range_is_rejected():
    with pytest.raises(ValueError, match="ephemeral"):
        attacks.PortScanConfig(source_port=1024)


def test_a_source_port_past_the_field_width_is_rejected():
    with pytest.raises(ValueError, match="ephemeral"):
        attacks.PortScanConfig(source_port=70000)


def test_generate_port_scan_places_every_probe_inside_the_baseline():
    config = attacks.PortScanConfig(offset_s=10.0, duration_s=5.0, port_count=20)
    frame = attacks.generate_port_scan(BASELINE, config)

    probes = frame[frame["label"] == "port_scan"]
    assert len(probes) == config.port_count
    assert probes["dst_port"].nunique() == config.port_count


def test_a_ddos_protocol_outside_tcp_or_udp_is_rejected():
    with pytest.raises(ValueError, match="tcp or udp"):
        attacks.DDoSConfig(protocol="icmp")


def test_a_ddos_target_port_past_the_field_width_is_rejected():
    with pytest.raises(ValueError, match="target port"):
        attacks.DDoSConfig(target_port=70000)


def test_a_ddos_with_no_attacking_hosts_is_rejected():
    with pytest.raises(ValueError, match="host_count"):
        attacks.DDoSConfig(host_count=0)


def test_a_ddos_packet_count_below_one_per_host_is_rejected():
    with pytest.raises(ValueError, match="one packet each"):
        attacks.DDoSConfig(host_count=100, packet_count=50)


def test_flood_sources_returns_one_address_per_attacking_host():
    sources = attacks.flood_sources(attacks.DDoSConfig(host_count=50))

    assert len(sources) == 50


def test_flood_sources_are_distinct_within_the_documentation_range():
    sources = attacks.flood_sources(attacks.DDoSConfig(host_count=200))

    assert len(set(sources.tolist())) == 200


def test_flood_sources_repeat_past_the_documentation_range():
    sources = attacks.flood_sources(attacks.DDoSConfig(host_count=300, packet_count=300))

    assert sources[0] == sources[254]


def test_flood_times_sends_one_arrival_per_packet():
    config = attacks.DDoSConfig(host_count=10, packet_count=500, duration_s=5.0)

    assert len(attacks.flood_times(config, START)) == 500


def test_flood_times_run_on_an_even_grid_from_the_window_start():
    config = attacks.DDoSConfig(host_count=4, packet_count=4, duration_s=8.0)
    times = attacks.flood_times(config, START)

    assert times[0] == START
    assert list(times[1:] - times[:-1]) == [pd.Timedelta(seconds=2.0)] * 3


def test_the_last_flood_packet_lands_before_the_window_closes():
    config = attacks.DDoSConfig(host_count=10, packet_count=10, duration_s=5.0)
    times = attacks.flood_times(config, START)

    assert times.max() < START + pd.Timedelta(seconds=5.0)


def test_a_flood_frame_addresses_every_packet_to_the_target():
    config = attacks.DDoSConfig(host_count=4, packet_count=8, duration_s=1.0)
    frame = attacks.flood_frame(config, START)

    assert (frame["dst_ip"] == config.target).all()
    assert (frame["dst_port"] == config.target_port).all()
    assert (frame["protocol"] == config.protocol).all()


def test_a_flood_frame_cycles_source_hosts_round_robin():
    config = attacks.DDoSConfig(host_count=4, packet_count=8, duration_s=1.0)
    frame = attacks.flood_frame(config, START)
    sources = list(attacks.flood_sources(config))

    assert list(frame["src_ip"])[:4] == sources
    assert list(frame["src_ip"])[4:] == sources


def test_a_tcp_flood_carries_a_syn_flag_and_a_udp_flood_carries_none():
    tcp_config = attacks.DDoSConfig(protocol="tcp", host_count=2, packet_count=4, duration_s=1.0)
    udp_config = attacks.DDoSConfig(protocol="udp", host_count=2, packet_count=4, duration_s=1.0)

    tcp_frame = attacks.flood_frame(tcp_config, START)
    udp_frame = attacks.flood_frame(udp_config, START)

    assert (tcp_frame["tcp_flags"] == "S").all()
    assert udp_frame["tcp_flags"].isna().all()
    assert (tcp_frame["label"] == "ddos").all()


def test_generate_ddos_burst_places_every_packet_inside_the_baseline():
    config = attacks.DDoSConfig(offset_s=10.0, duration_s=5.0, host_count=10, packet_count=50)
    frame = attacks.generate_ddos_burst(BASELINE, config)

    flood = frame[frame["label"] == "ddos"]
    assert len(flood) == config.packet_count
    assert flood["src_ip"].nunique() == config.host_count


def test_a_beacon_c2_port_past_the_field_width_is_rejected():
    with pytest.raises(ValueError, match="c2 port"):
        attacks.BeaconConfig(c2_port=70000)


def test_a_beacon_with_a_non_positive_interval_is_rejected():
    with pytest.raises(ValueError, match="interval_s"):
        attacks.BeaconConfig(interval_s=0.0)


def test_a_beacon_jitter_at_or_above_the_interval_is_rejected():
    with pytest.raises(ValueError, match="jitter_s"):
        attacks.BeaconConfig(interval_s=30.0, jitter_s=30.0)


def test_a_beacon_with_no_checkins_is_rejected():
    with pytest.raises(ValueError, match="beacon_count"):
        attacks.BeaconConfig(beacon_count=0)
