import pandas as pd

from icsad.protocols.modbus import decode_modbus_events_from_packets


def test_decode_modbus_events_empty_input():
    df_packets = pd.DataFrame(columns=["ts","src_ip","dst_ip","src_port","dst_port","payload_hex"])
    df_events = decode_modbus_events_from_packets(df_packets)
    assert "function_code" in df_events.columns
    assert len(df_events) == 0


def test_decode_modbus_events_filters_by_port():
    # One Modbus-ish payload with dst_port 502 should be considered;
    # another packet without port 502 should be ignored.
    # MBAP: tid=0x0001 pid=0x0000 len=0x0006 uid=0xff
    # PDU: fc=0x03, addr=0x0063, qty=0x001e  (common read holding registers)
    payload_hex = "000100000006ff030063001e"

    df_packets = pd.DataFrame(
        [
            {"ts": 1.0, "src_ip": "1.1.1.1", "dst_ip": "2.2.2.2", "src_port": 12345, "dst_port": 502, "payload_hex": payload_hex},
            {"ts": 2.0, "src_ip": "1.1.1.1", "dst_ip": "2.2.2.2", "src_port": 1111, "dst_port": 2222, "payload_hex": payload_hex},
        ]
    )

    df_events = decode_modbus_events_from_packets(df_packets)
    assert len(df_events) == 1
    row = df_events.iloc[0]
    assert int(row["function_code"]) == 0x03
    assert row["direction"] == "request"
    assert int(row["address"]) == 0x0063
    assert int(row["quantity"]) == 0x001e
