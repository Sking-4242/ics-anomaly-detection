from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

Label = Literal["suspicious_high", "suspicious_med", "benign_like", "unknown"]


@dataclass(frozen=True)
class LabelConfig:
    window_seconds: int = 5

    # Timing-first triggers
    dt_min_tight_high: float = 0.00010  # 100 microseconds
    dt_min_tight_med: float = 0.00020   # 200 microseconds
    min_events_for_timing: int = 10

    # Secondary (semantics) boosters
    unique_fc_hi: int = 8
    exception_rate_hi: float = 0.05

    # "Benign-like" conditions (conservative)
    benign_min_events: int = 10
    benign_dt_min_floor: float = 0.005  # 5ms
    benign_dt_max_ceiling: float = 1.0


def label_windows_timing_first(df_windows: pd.DataFrame, cfg: LabelConfig = LabelConfig()) -> pd.DataFrame:
    """
    Produce heuristic labels for evaluation.
    This is NOT ground truth. It's a defensible proxy:
      - suspicious_high: strong timing burst + enough volume, optionally strengthened by semantics
      - suspicious_med: moderate timing burst + enough volume, optionally strengthened by semantics
      - benign_like: stable timing, no exceptions, low semantic variance
      - unknown: everything else

    Expected df_windows columns (from session_features canonical key build):
      client_ip, server_ip, client_port, server_port, window_start,
      event_count, exception_rate, unique_function_codes, dt_min, dt_max
    """
    required = {
        "client_ip","server_ip","client_port","server_port","window_start",
        "event_count","exception_rate","unique_function_codes","dt_min","dt_max"
    }
    missing = required - set(df_windows.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df_windows.copy()

    # Normalize numeric columns
    for c in ["event_count", "unique_function_codes", "server_port"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    for c in ["exception_rate", "dt_min", "dt_max", "window_start"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

    df["client_port"] = pd.to_numeric(df["client_port"], errors="coerce")

    # Helper booleans
    enough_volume = df["event_count"] >= cfg.min_events_for_timing
    timing_high = (df["dt_min"] > 0) & (df["dt_min"] <= cfg.dt_min_tight_high)
    timing_med = (df["dt_min"] > 0) & (df["dt_min"] <= cfg.dt_min_tight_med)

    sem_boost = (df["unique_function_codes"] >= cfg.unique_fc_hi) | (df["exception_rate"] >= cfg.exception_rate_hi)

    benign_like = (
        (df["event_count"] >= cfg.benign_min_events)
        & (df["exception_rate"] == 0.0)
        & (df["unique_function_codes"] < cfg.unique_fc_hi)
        & (df["dt_min"] >= cfg.benign_dt_min_floor)
        & (df["dt_max"] <= cfg.benign_dt_max_ceiling)
    )

    labels: list[Label] = []
    label_reason: list[str] = []

    for row in df.itertuples(index=False):
        ev = int(getattr(row, "event_count"))
        dt_min = float(getattr(row, "dt_min"))
        dt_max = float(getattr(row, "dt_max"))
        exc = float(getattr(row, "exception_rate"))
        ufc = int(getattr(row, "unique_function_codes"))

        # benign-like first (conservative)
        if benign_like.loc[row.Index] if hasattr(row, "Index") else False:
            labels.append("benign_like")
            label_reason.append(f"stable_timing+no_exceptions (dt_min={dt_min:.4f}, dt_max={dt_max:.3f}, ufc={ufc})")
            continue

        # suspicious (timing-first)
        is_enough = ev >= cfg.min_events_for_timing
        is_t_high = (dt_min > 0) and (dt_min <= cfg.dt_min_tight_high)
        is_t_med = (dt_min > 0) and (dt_min <= cfg.dt_min_tight_med)

        if is_enough and is_t_high:
            # high confidence; semantics can strengthen the reason text
            if (ufc >= cfg.unique_fc_hi) or (exc >= cfg.exception_rate_hi):
                labels.append("suspicious_high")
                label_reason.append(f"tight_burst+sem_boost (dt_min={dt_min:.6f}, ev={ev}, ufc={ufc}, exc={exc:.3f})")
            else:
                labels.append("suspicious_high")
                label_reason.append(f"tight_burst (dt_min={dt_min:.6f}, ev={ev})")
            continue

        if is_enough and is_t_med:
            if (ufc >= cfg.unique_fc_hi) or (exc >= cfg.exception_rate_hi):
                labels.append("suspicious_med")
                label_reason.append(f"burst+sem_boost (dt_min={dt_min:.6f}, ev={ev}, ufc={ufc}, exc={exc:.3f})")
            else:
                labels.append("suspicious_med")
                label_reason.append(f"burst (dt_min={dt_min:.6f}, ev={ev})")
            continue

        # unknown
        labels.append("unknown")
        label_reason.append(f"no_rule_match (dt_min={dt_min:.6f}, ev={ev}, ufc={ufc}, exc={exc:.3f})")

    # The loop above can't use row.Index reliably; compute benign_like separately and overwrite where True
    df["label"] = labels
    df["label_reason"] = label_reason
    df.loc[benign_like, "label"] = "benign_like"
    df.loc[benign_like, "label_reason"] = (
        "stable_timing+no_exceptions (dt_min>=%.4f, dt_max<=%.1f)"
        % (cfg.benign_dt_min_floor, cfg.benign_dt_max_ceiling)
    )

    # Convenience: binary target for evaluation (high-confidence suspicious)
    df["is_suspicious_high"] = (df["label"] == "suspicious_high").astype(int)
    df["is_suspicious_any"] = df["label"].isin(["suspicious_high", "suspicious_med"]).astype(int)

    # Keep compact output
    keep = [
        "client_ip","server_ip","client_port","server_port","window_start",
        "label","label_reason","is_suspicious_high","is_suspicious_any",
        "event_count","dt_min","dt_max","exception_rate","unique_function_codes",
    ]
    return df[keep]
