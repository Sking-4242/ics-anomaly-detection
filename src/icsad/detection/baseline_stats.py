from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from icsad.explain.rationale import Rationale, fmt_rationale


@dataclass(frozen=True)
class Alert:
    client_ip: str
    server_ip: str
    client_port: int | None
    server_port: int
    window_start: float

    severity: float
    reasons: list[str]

    event_count: int
    request_count: int
    response_count: int
    exception_rate: float
    unique_function_codes: int

    dt_min: float
    dt_max: float
    dt_mean: float
    request_response_ratio: float | None


def _zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    mu = float(x.mean()) if len(x) else 0.0
    sigma = float(x.std(ddof=0)) if len(x) else 0.0
    if sigma == 0.0:
        return pd.Series([0.0] * len(x), index=x.index)
    return (x - mu) / sigma


def detect_baseline_alerts(
    df_windows: pd.DataFrame,
    *,
    z_event_count: float = 4.0,
    exception_rate_hi: float = 0.05,
    unique_fc_hi: int = 8,
    dt_min_lo: float = 0.0002,
    dt_max_hi: float = 2.0,
    rr_ratio_hi: float = 20.0,
) -> pd.DataFrame:
    required = {
        "client_ip","server_ip","client_port","server_port","window_start",
        "event_count","request_count","response_count","exception_rate","unique_function_codes",
        "dt_min","dt_max","dt_mean","request_response_ratio",
    }
    missing = required - set(df_windows.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df_windows.copy()

    for c in ["event_count", "request_count", "response_count", "unique_function_codes", "server_port"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    for c in ["exception_rate", "dt_min", "dt_max", "dt_mean", "window_start"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

    df["client_port"] = pd.to_numeric(df["client_port"], errors="coerce")
    df["request_response_ratio"] = pd.to_numeric(df["request_response_ratio"], errors="coerce")

    df["event_count_z"] = _zscore(df["event_count"])

    alerts: list[Alert] = []

    for row in df.itertuples(index=False):
        reasons: list[Rationale] = []
        severity = 0.0

        event_count = int(getattr(row, "event_count"))
        request_count = int(getattr(row, "request_count"))
        response_count = int(getattr(row, "response_count"))
        exception_rate = float(getattr(row, "exception_rate"))
        unique_fcs = int(getattr(row, "unique_function_codes"))
        dt_min = float(getattr(row, "dt_min"))
        dt_max = float(getattr(row, "dt_max"))
        dt_mean = float(getattr(row, "dt_mean"))
        rr_ratio = getattr(row, "request_response_ratio")
        rr_ratio_val = float(rr_ratio) if rr_ratio is not None and not (isinstance(rr_ratio, float) and np.isnan(rr_ratio)) else None
        event_count_z = float(getattr(row, "event_count_z"))

        if event_count_z >= z_event_count:
            severity += min(10.0, event_count_z)
            reasons.append(Rationale("VOL_Z", "Unusually high Modbus event volume for this window",
                                     {"event_count": event_count, "z": round(event_count_z, 2), "threshold": z_event_count}))

        if exception_rate >= exception_rate_hi and event_count >= 5:
            severity += 6.0
            reasons.append(Rationale("EXC_RATE", "High exception rate suggests errors, probing, or misconfiguration",
                                     {"exception_rate": round(exception_rate, 4), "threshold": exception_rate_hi, "event_count": event_count}))

        if unique_fcs >= unique_fc_hi and event_count >= 10:
            severity += 4.0
            reasons.append(Rationale("FC_DIVERSITY", "Unusually high function-code diversity in a short window",
                                     {"unique_function_codes": unique_fcs, "threshold": unique_fc_hi, "event_count": event_count}))

        if dt_min > 0 and dt_min <= dt_min_lo and event_count >= 10:
            severity += 3.0
            reasons.append(Rationale("DT_MIN", "Very small inter-arrival time (burst) may indicate automation/replay",
                                     {"dt_min": round(dt_min, 6), "threshold": dt_min_lo}))

        if dt_max >= dt_max_hi and event_count >= 3:
            severity += 2.0
            reasons.append(Rationale("DT_MAX", "Large inter-arrival time suggests irregular traffic or low-and-slow behavior",
                                     {"dt_max": round(dt_max, 3), "threshold": dt_max_hi}))

        # Request/response imbalance – now meaningful with canonical flow keys
        if response_count == 0 and request_count >= 10:
            severity += 5.0
            reasons.append(Rationale("NO_RESP", "Many requests with zero responses in window",
                                     {"request_count": request_count, "response_count": response_count}))
        elif rr_ratio_val is not None and rr_ratio_val >= rr_ratio_hi and request_count >= 10:
            severity += 3.0
            reasons.append(Rationale("RR_IMBAL", "Request/response ratio is unusually high",
                                     {"request_response_ratio": round(rr_ratio_val, 2), "threshold": rr_ratio_hi}))

        if reasons:
            alerts.append(
                Alert(
                    client_ip=str(getattr(row, "client_ip")),
                    server_ip=str(getattr(row, "server_ip")),
                    client_port=int(getattr(row, "client_port")) if getattr(row, "client_port") is not None and not np.isnan(getattr(row, "client_port")) else None,
                    server_port=int(getattr(row, "server_port")),
                    window_start=float(getattr(row, "window_start")),
                    severity=float(severity),
                    reasons=[fmt_rationale(r) for r in reasons],
                    event_count=event_count,
                    request_count=request_count,
                    response_count=response_count,
                    exception_rate=exception_rate,
                    unique_function_codes=unique_fcs,
                    dt_min=dt_min,
                    dt_max=dt_max,
                    dt_mean=dt_mean,
                    request_response_ratio=rr_ratio_val,
                )
            )

    out = pd.DataFrame([asdict(a) for a in alerts])
    if out.empty:
        return pd.DataFrame()

    return out.sort_values(["severity", "window_start"], ascending=[False, True]).reset_index(drop=True)
