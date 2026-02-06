from __future__ import annotations

import math
import pandas as pd

from icsad.features.base import WindowSpec

MODBUS_SERVER_PORT = 502


def _window_start(ts: float, win_s: int) -> float:
    return math.floor(ts / win_s) * win_s


def _canonicalize_flow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create canonical client/server endpoints so requests/responses aggregate together.

    For Modbus TCP:
      - server_port is 502
      - client_port is the non-502 port
      - client_ip is the IP on the client_port side
      - server_ip is the IP on the 502 side
    """
    df = df.copy()

    # Identify which side is server based on port 502
    src_is_server = df["src_port"] == MODBUS_SERVER_PORT
    dst_is_server = df["dst_port"] == MODBUS_SERVER_PORT

    # server_ip/port
    df["server_ip"] = df["src_ip"].where(src_is_server, df["dst_ip"].where(dst_is_server, pd.NA))
    df["server_port"] = MODBUS_SERVER_PORT

    # client_ip/port
    df["client_ip"] = df["dst_ip"].where(src_is_server, df["src_ip"].where(dst_is_server, pd.NA))
    df["client_port"] = df["dst_port"].where(src_is_server, df["src_port"].where(dst_is_server, pd.NA))

    # Drop rows where we can't determine client/server (should be rare after port filtering)
    df = df[df["client_ip"].notna() & df["server_ip"].notna() & df["client_port"].notna()].copy()

    # Ensure ints
    df["client_port"] = pd.to_numeric(df["client_port"], errors="coerce").astype("Int64")

    return df


def build_modbus_window_features(df_modbus: pd.DataFrame, window: WindowSpec = WindowSpec()) -> pd.DataFrame:
    required = {
        "ts", "src_ip", "dst_ip", "src_port", "dst_port",
        "direction", "function_code", "is_exception", "address", "quantity",
    }
    missing = required - set(df_modbus.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df_modbus.copy()

    # Types
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df = df[df["ts"].notna()].copy()

    df["src_port"] = pd.to_numeric(df["src_port"], errors="coerce").astype("Int64")
    df["dst_port"] = pd.to_numeric(df["dst_port"], errors="coerce").astype("Int64")

    df["function_code"] = pd.to_numeric(df["function_code"], errors="coerce").fillna(-1).astype(int)
    df["is_exception"] = df["is_exception"].fillna(False).astype(bool)

    df["is_request"] = (df["direction"] == "request").astype(int)
    df["is_response"] = (df["direction"] == "response").astype(int)

    df["address"] = pd.to_numeric(df["address"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    # Filter to modbus port 502 traffic only (defensive)
    df = df[(df["src_port"] == MODBUS_SERVER_PORT) | (df["dst_port"] == MODBUS_SERVER_PORT)].copy()

    # Canonicalize flow so req/resp aggregate together
    df = _canonicalize_flow(df)

    win_s = int(window.seconds)
    df["window_start"] = df["ts"].apply(lambda x: _window_start(float(x), win_s)).astype(float)

    keys = ["client_ip", "server_ip", "client_port", "server_port", "window_start"]

    # Timing features within group
    df_sorted = df.sort_values(keys + ["ts"]).copy()
    df_sorted["dt"] = df_sorted.groupby(keys)["ts"].diff()

    agg = df_sorted.groupby(keys).agg(
        event_count=("ts", "size"),
        request_count=("is_request", "sum"),
        response_count=("is_response", "sum"),
        exception_count=("is_exception", "sum"),
        unique_function_codes=("function_code", "nunique"),
        req_address_nunique=("address", lambda s: s.dropna().nunique()),
        req_address_min=("address", "min"),
        req_address_max=("address", "max"),
        req_quantity_mean=("quantity", "mean"),
        req_quantity_max=("quantity", "max"),
        dt_mean=("dt", "mean"),
        dt_min=("dt", "min"),
        dt_max=("dt", "max"),
    ).reset_index()

    agg["exception_rate"] = agg["exception_count"] / agg["event_count"].clip(lower=1)
    agg["request_response_ratio"] = agg["request_count"] / agg["response_count"].replace(0, pd.NA)

    # Fill NaNs for numeric columns (except request_response_ratio, keep NA meaningful)
    fill_zero_cols = [
        "dt_mean", "dt_min", "dt_max",
        "req_address_nunique", "req_address_min", "req_address_max",
        "req_quantity_mean", "req_quantity_max",
    ]
    for c in fill_zero_cols:
        if c in agg.columns:
            agg[c] = agg[c].fillna(0)

    return agg
