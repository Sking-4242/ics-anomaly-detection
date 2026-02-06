from __future__ import annotations

from pathlib import Path

import pandas as pd

from icsad.detection.baseline_stats import detect_baseline_alerts
from icsad.export.csv_export import export_alerts_csv


def main() -> None:
    win_path = Path("data/processed/modbus_window_features_5s.parquet")
    if not win_path.exists():
        raise SystemExit("Missing window features. Run: poetry run python scripts/build_features.py")

    df_win = pd.read_parquet(win_path)

    df_alerts = detect_baseline_alerts(df_win)

    out_csv = export_alerts_csv(df_alerts, "data/processed/alerts_baseline_5s.csv")
    out_parquet = Path("data/processed/alerts_baseline_5s.parquet")
    df_alerts.to_parquet(out_parquet, index=False)

    print(f"Wrote {out_csv} (rows={len(df_alerts)})")
    print(f"Wrote {out_parquet} (rows={len(df_alerts)})")

    if len(df_alerts) > 0:
        print(df_alerts.head(20))


if __name__ == "__main__":
    main()
