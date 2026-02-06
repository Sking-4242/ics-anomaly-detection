from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass(frozen=True)
class IFConfig:
    n_estimators: int = 200
    max_samples: str | int = "auto"
    contamination: float = 0.05  # expected anomaly fraction
    random_state: int = 42


def _feature_columns() -> list[str]:
    # Timing-first, semantics secondary
    return [
        "event_count",
        "dt_min",
        "dt_mean",
        "dt_max",
        "exception_rate",
        "unique_function_codes",
        "request_count",
        "response_count",
    ]


def train_iforest(df_windows: pd.DataFrame, cfg: IFConfig = IFConfig()) -> IsolationForest:
    cols = _feature_columns()
    missing = set(cols) - set(df_windows.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")

    X = df_windows[cols].copy()
    X = X.fillna(0.0)

    model = IsolationForest(
        n_estimators=cfg.n_estimators,
        max_samples=cfg.max_samples,
        contamination=cfg.contamination,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    model.fit(X)
    return model


def score_iforest(model: IsolationForest, df_windows: pd.DataFrame) -> pd.DataFrame:
    cols = _feature_columns()
    X = df_windows[cols].copy().fillna(0.0)

    # sklearn IF: higher score = more normal
    scores = model.score_samples(X)
    anomaly_score = -scores  # higher = more anomalous

    out = df_windows.copy()
    out["iforest_score"] = anomaly_score
    return out


def build_iforest_alerts(
    df_scored: pd.DataFrame,
    *,
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Convert IF scores into alerts.
    Choose either:
      - top_k windows by anomaly score, or
      - a fixed score_threshold
    """
    if top_k is None and score_threshold is None:
        raise ValueError("Provide either top_k or score_threshold")

    df = df_scored.copy().sort_values("iforest_score", ascending=False)

    if top_k is not None:
        df = df.head(top_k)

    if score_threshold is not None:
        df = df[df["iforest_score"] >= score_threshold]

    return df.reset_index(drop=True)
