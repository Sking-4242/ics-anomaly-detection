from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


def _top_counts(series: pd.Series, n: int = 5) -> list[dict[str, Any]]:
    vc = series.value_counts(dropna=True).head(n)
    return [{"value": (int(i) if isinstance(i, (int,)) else i), "count": int(c)} for i, c in vc.items()]


def enrich_alerts_with_evidence(
    df_alerts: pd.DataFrame,
    df_modbus_events: pd.DataFrame,
    window_seconds: int = 5,
    sample_events: int = 8,
) -> pd.DataFrame:
    """
    Adds evidence fields to each alert:
      - top_function_codes
      - top_addresses
      - top_quantities
      - exception_summary
      - sample_event_rows (compact list)

    Requires canonical alert keys:
      client_ip, server_ip, client_port, server_port, window_start
    """
    required_alert_cols = {"client_ip", "server_ip", "client_port", "server_port", "window_start"}
    missing_a = required_alert_cols - set(df_alerts.columns)
    if missing_a:
        raise ValueError(f"Alerts missing columns: {sorted(missing_a)}")

    required_evt_cols = {
        "ts", "src_ip", "dst_ip", "src_port", "dst_port",
        "transaction_id", "unit_id", "function_code", "is_exception", "exception_code",
        "direction", "address", "quantity", "raw_len",
    }
    missing_e = required_evt_cols - set(df_modbus_events.columns)
    if missing_e:
        raise ValueError(f"Modbus events missing columns: {sorted(missing_e)}")

    ev = df_modbus_events.copy()

    # Determine canonical client/server columns for events using port 502 logic
    # server side is port 502
    ev["src_port"] = pd.to_numeric(ev["src_port"], errors="coerce")
    ev["dst_port"] = pd.to_numeric(ev["dst_port"], errors="coerce")

    src_is_server = ev["src_port"] == 502
    dst_is_server = ev["dst_port"] == 502

    ev["server_ip"] = ev["src_ip"].where(src_is_server, ev["dst_ip"].where(dst_is_server, pd.NA))
    ev["client_ip"] = ev["dst_ip"].where(src_is_server, ev["src_ip"].where(dst_is_server, pd.NA))
    ev["server_port"] = 502
    ev["client_port"] = ev["dst_port"].where(src_is_server, ev["src_port"].where(dst_is_server, pd.NA))

    ev = ev[ev["client_ip"].notna() & ev["server_ip"].notna() & ev["client_port"].notna()].copy()

    ev["client_port"] = pd.to_numeric(ev["client_port"], errors="coerce")
    ev["ts"] = pd.to_numeric(ev["ts"], errors="coerce")

    # Window start
    ev["window_start"] = (ev["ts"] // window_seconds) * window_seconds

    out_rows: list[dict[str, Any]] = []

    for a in df_alerts.itertuples(index=False):
        client_ip = str(getattr(a, "client_ip"))
        server_ip = str(getattr(a, "server_ip"))
        client_port = getattr(a, "client_port")
        server_port = int(getattr(a, "server_port"))
        window_start = float(getattr(a, "window_start"))

        # Filter to matching window + flow
        mask = (
            (ev["client_ip"] == client_ip)
            & (ev["server_ip"] == server_ip)
            & (ev["server_port"] == server_port)
            & (ev["window_start"] == window_start)
        )

        # client_port can be float/None due to parquet typing; compare carefully
        if client_port is None or (isinstance(client_port, float) and pd.isna(client_port)):
            # If client_port missing in alert, don't filter it
            pass
        else:
            mask = mask & (ev["client_port"] == float(client_port))

        slice_ev = ev[mask].sort_values("ts")

        # Evidence summaries
        top_fcs = _top_counts(slice_ev["function_code"], n=5) if len(slice_ev) else []
        top_addrs = _top_counts(slice_ev[slice_ev["direction"] == "request"]["address"].dropna(), n=5) if len(slice_ev) else []
        top_qty = _top_counts(slice_ev[slice_ev["direction"] == "request"]["quantity"].dropna(), n=5) if len(slice_ev) else []

        exc = slice_ev[slice_ev["is_exception"] == True]
        exc_summary = []
        if len(exc):
            exc_summary = _top_counts(exc["exception_code"], n=5)

        # Sample rows (compact)
        samp = slice_ev.head(sample_events)
        sample = []
        for r in samp.itertuples(index=False):
            sample.append(
                {
                    "ts": float(getattr(r, "ts")),
                    "dir": str(getattr(r, "direction")),
                    "tid": int(getattr(r, "transaction_id")),
                    "uid": int(getattr(r, "unit_id")),
                    "fc": int(getattr(r, "function_code")),
                    "addr": (None if pd.isna(getattr(r, "address")) else int(getattr(r, "address"))),
                    "qty": (None if pd.isna(getattr(r, "quantity")) else int(getattr(r, "quantity"))),
                    "exc": (None if pd.isna(getattr(r, "exception_code")) else int(getattr(r, "exception_code"))),
                    "len": int(getattr(r, "raw_len")),
                }
            )

        row_dict = a._asdict() if hasattr(a, "_asdict") else dict(a)
        row_dict.update(
            {
                "e_top_function_codes": top_fcs,
                "e_top_addresses": top_addrs,
                "e_top_quantities": top_qty,
                "e_exception_codes": exc_summary,
                "e_sample_events": sample,
                "e_event_count_window": int(len(slice_ev)),
            }
        )
        out_rows.append(row_dict)

    return pd.DataFrame(out_rows)
