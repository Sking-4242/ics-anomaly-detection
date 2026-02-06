from __future__ import annotations

from pathlib import Path

import pandas as pd

from icsad.detection.hybrid import build_hybrid_alerts
from icsad.detection.iforest import IFConfig, train_iforest, score_iforest
from icsad.evaluation.metrics import evaluate_alerts_against_labels
from icsad.export.csv_export import export_alerts_csv


def main() -> None:
    win_path = Path("data/processed/modbus_window_features_5s.parquet")
    alerts_base_path = Path("data/processed/alerts_baseline_5s.parquet")
    labels_path = Path("data/processed/window_labels_5s.parquet")

    if not win_path.exists():
        raise SystemExit("Missing window features. Run: poetry run python scripts/build_features.py")
    if not alerts_base_path.exists():
        raise SystemExit("Missing baseline alerts. Run: poetry run python scripts/run_baseline.py")
    if not labels_path.exists():
        raise SystemExit("Missing labels. Run: poetry run python scripts/run_evaluation.py")

    df_win = pd.read_parquet(win_path)
    df_base = pd.read_parquet(alerts_base_path)
    df_labels = pd.read_parquet(labels_path)

    # Train/score IF on all windows (unsupervised)
    cfg = IFConfig(contamination=0.05)
    model = train_iforest(df_win, cfg)
    df_scored = score_iforest(model, df_win)

    # Hybrid ranking (baseline explainability + IF ranking)
    alpha, beta = 0.6, 0.4
    df_hybrid = build_hybrid_alerts(df_base, df_scored, alpha=alpha, beta=beta, top_k=50)

    out_parquet = Path("data/processed/alerts_hybrid_5s.parquet")
    df_hybrid.to_parquet(out_parquet, index=False)

    out_csv = export_alerts_csv(df_hybrid, "data/processed/alerts_hybrid_5s.csv")

    # Evaluate hybrid ranking using same label harness
    ranked, summary = evaluate_alerts_against_labels(
        df_hybrid.rename(columns={"hybrid_score": "severity_hybrid"}),
        df_labels,
        severity_col="severity_hybrid",
    )
    out_eval = Path("data/processed/eval_hybrid_5s.csv")
    ranked.to_csv(out_eval, index=False)

    print(f"Wrote {out_parquet} (rows={len(df_hybrid)})")
    print(f"Wrote {out_csv} (rows={len(df_hybrid)})")
    print(f"Wrote {out_eval} (rows={len(ranked)})\n")

    print("=== Hybrid Evaluation (timing-first labels) ===")
    print(f"windows_total:         {summary.windows_total}")
    print(f"alerts_total:          {summary.alerts_total}")
    print(f"alert_rate:            {summary.alert_rate:.3f}")
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
        "severity_hybrid", "severity", "iforest_score",
        "client_ip", "server_ip", "client_port", "window_start",
        "label", "is_suspicious_any", "reasons",
    ]
    present = [c for c in show_cols if c in ranked.columns]
    print(ranked[present].head(15))


if __name__ == "__main__":
    main()
