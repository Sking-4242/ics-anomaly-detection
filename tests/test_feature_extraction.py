import pandas as pd

from icsad.features.base import WindowSpec
from icsad.features.packet_features import build_modbus_packet_features
from icsad.features.session_features import build_modbus_window_features


def test_window_spec_validation():
    try:
        WindowSpec(seconds=0)
        assert False, "expected ValueError"
    except ValueError:
        assert True


def test_packet_features_smoke():
    df = pd.DataFrame(
        [
            {
                "ts": 1.0,
                "src_ip": "1.1.1.1",
                "dst_ip": "2.2.2.2",
                "src_port": 12345,
                "dst_port": 502,
                "function_code": 3,
                "is_exception": False,
                "direction": "request",
                "address": 99,
                "quantity": 30,
                "value": None,
                "raw_len": 12,
            }
        ]
    )
    out = build_modbus_packet_features(df)
    assert len(out) == 1
    assert "is_request" in out.columns
    assert int(out.iloc[0]["is_request"]) == 1


def test_window_features_grouping_5s():
    df = pd.DataFrame(
        [
            # same flow, same 5s window
            {"ts": 10.1, "src_ip": "a", "dst_ip": "b", "src_port": 111, "dst_port": 502, "direction": "request", "function_code": 3, "is_exception": False, "address": 10, "quantity": 2},
            {"ts": 12.2, "src_ip": "a", "dst_ip": "b", "src_port": 111, "dst_port": 502, "direction": "request", "function_code": 3, "is_exception": False, "address": 10, "quantity": 2},
            # next window
            {"ts": 15.0, "src_ip": "a", "dst_ip": "b", "src_port": 111, "dst_port": 502, "direction": "response", "function_code": 3, "is_exception": False, "address": None, "quantity": None},
        ]
    )
    out = build_modbus_window_features(df, window=WindowSpec(seconds=5))
    # Expect 2 groups: [10-15) and [15-20)
    assert len(out) == 2
    assert out["event_count"].sum() == 3
