from __future__ import annotations

from pathlib import Path

import pandas as pd

from icsad.detection.iforest import IFConfig, train_iforest, score_iforest, build_iforest_alerts
from icsad.evaluation.metrics import evaluate_alerts_against_labels


def main() -> None:
    win_path = Path("data/processed/modbus_window_features_5s.parquet")
    labels_path = Path("data/processed/window_labels_5s.parquet")

    if not win_path.exists():
        raise SystemExit("Missing window features. Run: poetry run python scripts/build_features.py")
    if not labels_path.exists():
        raise SystemExit("Missing labels. Run: poetry run python scripts/run_evaluation.py")

    df_win = pd.read_parquet(win_path)
    df_labels = pd.read_parquet(labels_path)

    # 1) Train IF on all windows (unsupervised)
    cfg = IFConfig(contamination=0.05)
    model = train_iforest(df_win, cfg)

    # 2) Score windows
    df_scored = score_iforest(model, df_win)

    # 3) Build alerts (top-K strategy)
    top_k = 50
    df_alerts_if = build_iforest_alerts(df_scored, top_k=top_k)

    # Keep only alert-relevant columns + score
    keep = [
        "client_ip","server_ip","client_port","server_port","window_start",
        "event_count","request_count","response_count",
        "exception_rate","unique_function_codes",
        "dt_min","dt_mean","dt_max",
        "iforest_score",
    ]
    df_alerts_if = df_alerts_if[keep]

    # 4) Evaluate
    ranked, summary = evaluate_alerts_against_labels(
        df_alerts_if.rename(columns={"iforest_score": "severity"}),
        df_labels,
        severity_col="severity",
    )

    out_csv = Path("data/processed/eval_iforest_top50_5s.csv")
    ranked.to_csv(out_csv, index=False)

    print(f"Wrote {out_csv} (rows={len(ranked)})\n")

    print("=== Isolation Forest Evaluation (top 50 windows) ===")
    print(f"windows_total:         {summary.windows_total}")
    print(f"alerts_total:          {summary.alerts_total}")
    print(f"alert_rate:            {summary.alert_rate:.3f}")
    print(f"suspicious_high_total: {summary.suspicious_high_total}")
    print(f"suspicious_any_total:  {summary.suspicious_any_total}")
    print("")
    print(f"precision@10  (high):  {summary.precision_at_10_high}")
    print(f"precision@25  (high):  {summary.precision_at_25_high}")
    print(f"precision@50  (high):  {summary.precision_at_50_high}")
    print("")
    print(f"precision@10  (any):   {summary.precision_at_10_any}")
    print(f"precision@25  (any):   {summary.precision_at_25_any}")
    print(f"precision@50  (any):   {summary.precision_at_50_any}")
    print("")

    show_cols = [
        "severity","client_ip","server_ip","client_port","window_start",
        "is_suspicious_high","is_suspicious_any",
    ]
    present = [c for c in show_cols if c in ranked.columns]
    print(ranked[present].head(15))


if __name__ == "__main__":
    main()
