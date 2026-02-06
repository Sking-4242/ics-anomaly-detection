from __future__ import annotations

import pandas as pd


def build_modbus_packet_features(df_modbus: pd.DataFrame) -> pd.DataFrame:
    """
    Packet-level (event-level) features derived from decoded Modbus events.
    Keeps features explainable and simple.

    Expected columns (from modbus_events.parquet):
      ts, src_ip, dst_ip, src_port, dst_port,
      function_code, is_exception, exception_code,
      direction, address, quantity, value, raw_len
    """
    required = {
        "ts",
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "function_code",
        "is_exception",
        "direction",
        "address",
        "quantity",
        "value",
        "raw_len",
    }
    missing = required - set(df_modbus.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df_modbus.copy()

    # Normalize types / missing values
    df["function_code"] = pd.to_numeric(df["function_code"], errors="coerce").fillna(-1).astype(int)
    df["is_exception"] = df["is_exception"].fillna(False).astype(bool)
    df["address"] = pd.to_numeric(df["address"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["raw_len"] = pd.to_numeric(df["raw_len"], errors="coerce").fillna(0).astype(int)

    # Direction one-hot (explainable)
    df["is_request"] = (df["direction"] == "request").astype(int)
    df["is_response"] = (df["direction"] == "response").astype(int)

    # Cheap “semantic” features
    df["has_address"] = df["address"].notna().astype(int)
    df["has_quantity"] = df["quantity"].notna().astype(int)

    # Keep metadata + numeric features
    cols = [
        "ts",
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "function_code",
        "is_exception",
        "is_request",
        "is_response",
        "address",
        "quantity",
        "value",
        "raw_len",
        "has_address",
        "has_quantity",
    ]
    return df[cols]
