from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class EvalSummary:
    windows_total: int
    alerts_total: int
    alert_rate: float  # alerts / windows

    suspicious_high_total: int
    suspicious_any_total: int

    precision_at_10_high: float | None
    precision_at_25_high: float | None
    precision_at_50_high: float | None

    precision_at_10_any: float | None
    precision_at_25_any: float | None
    precision_at_50_any: float | None


def _precision_at_k(df_ranked: pd.DataFrame, k: int, target_col: str) -> float | None:
    if len(df_ranked) == 0:
        return None
    top = df_ranked.head(k)
    if len(top) == 0:
        return None
    return float(top[target_col].sum() / len(top))


def evaluate_alerts_against_labels(
    df_alerts: pd.DataFrame,
    df_labels: pd.DataFrame,
    *,
    severity_col: str = "severity",
) -> tuple[pd.DataFrame, EvalSummary]:
    """
    Join alerts with labels on canonical key and compute lightweight evaluation metrics.
    This is NOT ground truth evaluation; it measures alignment with our heuristic labels.
    """
    req_a = {"client_ip","server_ip","client_port","server_port","window_start", severity_col}
    req_l = {"client_ip","server_ip","client_port","server_port","window_start","is_suspicious_high","is_suspicious_any","label"}

    missing_a = req_a - set(df_alerts.columns)
    missing_l = req_l - set(df_labels.columns)
    if missing_a:
        raise ValueError(f"Alerts missing columns: {sorted(missing_a)}")
    if missing_l:
        raise ValueError(f"Labels missing columns: {sorted(missing_l)}")

    a = df_alerts.copy()
    l = df_labels.copy()

    # Normalize numeric ports/window
    a["client_port"] = pd.to_numeric(a["client_port"], errors="coerce")
    a["server_port"] = pd.to_numeric(a["server_port"], errors="coerce").fillna(0).astype(int)
    a["window_start"] = pd.to_numeric(a["window_start"], errors="coerce").fillna(0.0).astype(float)

    l["client_port"] = pd.to_numeric(l["client_port"], errors="coerce")
    l["server_port"] = pd.to_numeric(l["server_port"], errors="coerce").fillna(0).astype(int)
    l["window_start"] = pd.to_numeric(l["window_start"], errors="coerce").fillna(0.0).astype(float)

    joined = a.merge(
        l,
        on=["client_ip","server_ip","client_port","server_port","window_start"],
        how="left",
        suffixes=("", "_label"),
    )

    # If a window has an alert but no label record, treat as unknown
    joined["label"] = joined["label"].fillna("unknown")
    joined["is_suspicious_high"] = pd.to_numeric(joined["is_suspicious_high"], errors="coerce").fillna(0).astype(int)
    joined["is_suspicious_any"] = pd.to_numeric(joined["is_suspicious_any"], errors="coerce").fillna(0).astype(int)

    # Rank by severity
    ranked = joined.sort_values(severity_col, ascending=False).reset_index(drop=True)

    p10_h = _precision_at_k(ranked, 10, "is_suspicious_high")
    p25_h = _precision_at_k(ranked, 25, "is_suspicious_high")
    p50_h = _precision_at_k(ranked, 50, "is_suspicious_high")

    p10_a = _precision_at_k(ranked, 10, "is_suspicious_any")
    p25_a = _precision_at_k(ranked, 25, "is_suspicious_any")
    p50_a = _precision_at_k(ranked, 50, "is_suspicious_any")

    summary = EvalSummary(
        windows_total=int(len(df_labels)),
        alerts_total=int(len(df_alerts)),
        alert_rate=float(len(df_alerts) / max(1, len(df_labels))),
        suspicious_high_total=int(df_labels["is_suspicious_high"].sum()),
        suspicious_any_total=int(df_labels["is_suspicious_any"].sum()),
        precision_at_10_high=p10_h,
        precision_at_25_high=p25_h,
        precision_at_50_high=p50_h,
        precision_at_10_any=p10_a,
        precision_at_25_any=p25_a,
        precision_at_50_any=p50_a,
    )

    return ranked, summary
