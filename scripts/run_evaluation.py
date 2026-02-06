from __future__ import annotations

from pathlib import Path

import pandas as pd

from icsad.evaluation.labels import LabelConfig, label_windows_timing_first
from icsad.evaluation.metrics import evaluate_alerts_against_labels


def main() -> None:
    win_path = Path("data/processed/modbus_window_features_5s.parquet")
    alerts_path = Path("data/processed/alerts_baseline_5s.parquet")

    if not win_path.exists():
        raise SystemExit("Missing window features. Run: poetry run python scripts/build_features.py")
    if not alerts_path.exists():
        raise SystemExit("Missing baseline alerts. Run: poetry run python scripts/run_baseline.py")

    df_win = pd.read_parquet(win_path)
    df_alerts = pd.read_parquet(alerts_path)

    # 1) Label all windows (timing-first)
    cfg = LabelConfig(window_seconds=5)
    df_labels = label_windows_timing_first(df_win, cfg=cfg)

    out_labels = Path("data/processed/window_labels_5s.parquet")
    df_labels.to_parquet(out_labels, index=False)

    # 2) Evaluate alerts against labels
    ranked, summary = evaluate_alerts_against_labels(df_alerts, df_labels)

    out_eval = Path("data/processed/eval_baseline_5s.csv")
    ranked.to_csv(out_eval, index=False)

    print(f"Wrote {out_labels} (rows={len(df_labels)})")
    print(f"Wrote {out_eval} (rows={len(ranked)})\n")

    print("=== Evaluation Summary (heuristic labels, timing-first) ===")
    print(f"windows_total:         {summary.windows_total}")
    print(f"alerts_total:          {summary.alerts_total}")
    print(f"alert_rate:            {summary.alert_rate:.3f} alerts/window")
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

    # Show top 15 alerts with labels
    show_cols = [
        "severity","client_ip","server_ip","client_port","window_start",
        "label","is_suspicious_high","is_suspicious_any","reasons"
    ]
    present = [c for c in show_cols if c in ranked.columns]
    print(ranked[present].head(15))


if __name__ == "__main__":
    main()
