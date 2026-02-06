from __future__ import annotations

import pandas as pd


def _z(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    mu = float(x.mean()) if len(x) else 0.0
    sd = float(x.std(ddof=0)) if len(x) else 0.0
    if sd == 0.0:
        return pd.Series([0.0] * len(x), index=x.index)
    return (x - mu) / sd


def build_hybrid_alerts(
    df_baseline_alerts: pd.DataFrame,
    df_if_scored_windows: pd.DataFrame,
    *,
    alpha: float = 0.6,
    beta: float = 0.4,
    top_k: int | None = 50,
    score_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Hybrid ranking + gating.

    Ranking:
      hybrid_score = alpha * z(baseline_severity) + beta * z(iforest_score)

    Gating (choose one):
      - top_k: keep top K by hybrid_score (default 50)
      - score_threshold: keep hybrid_score >= threshold
      - if both provided, apply threshold then top_k

    Inputs:
      baseline alerts: client_ip, server_ip, client_port, server_port, window_start, severity, reasons, ...
      if scored windows: same key cols + iforest_score
    """
    required_a = {"client_ip","server_ip","client_port","server_port","window_start","severity","reasons"}
    required_w = {"client_ip","server_ip","client_port","server_port","window_start","iforest_score"}
    missing_a = required_a - set(df_baseline_alerts.columns)
    missing_w = required_w - set(df_if_scored_windows.columns)
    if missing_a:
        raise ValueError(f"Baseline alerts missing columns: {sorted(missing_a)}")
    if missing_w:
        raise ValueError(f"IF scored windows missing columns: {sorted(missing_w)}")

    a = df_baseline_alerts.copy()
    w = df_if_scored_windows.copy()

    # Normalize join key dtypes
    for df in (a, w):
        df["client_port"] = pd.to_numeric(df["client_port"], errors="coerce")
        df["server_port"] = pd.to_numeric(df["server_port"], errors="coerce").fillna(0).astype(int)
        df["window_start"] = pd.to_numeric(df["window_start"], errors="coerce").fillna(0.0).astype(float)

    merged = a.merge(
        w[["client_ip","server_ip","client_port","server_port","window_start","iforest_score"]],
        on=["client_ip","server_ip","client_port","server_port","window_start"],
        how="left",
    )

    merged["iforest_score"] = pd.to_numeric(merged["iforest_score"], errors="coerce").fillna(0.0)
    merged["severity_z"] = _z(merged["severity"])
    merged["iforest_z"] = _z(merged["iforest_score"])

    merged["hybrid_score"] = alpha * merged["severity_z"] + beta * merged["iforest_z"]
    merged["hybrid_alpha"] = alpha
    merged["hybrid_beta"] = beta

    merged = merged.sort_values("hybrid_score", ascending=False).reset_index(drop=True)

    if score_threshold is not None:
        merged = merged[merged["hybrid_score"] >= float(score_threshold)].copy()

    if top_k is not None:
        merged = merged.head(int(top_k)).copy()

    return merged.reset_index(drop=True)
