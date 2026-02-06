from __future__ import annotations

from pathlib import Path

import pandas as pd

from icsad.explain.report import enrich_alerts_with_evidence
from icsad.export.csv_export import export_alerts_csv


def main() -> None:
    alerts_path = Path("data/processed/alerts_hybrid_5s.parquet")
    events_path = Path("data/interim/modbus_events.parquet")

    if not alerts_path.exists():
        raise SystemExit("Missing hybrid alerts. Run: poetry run python scripts/run_hybrid.py")
    if not events_path.exists():
        raise SystemExit("Missing modbus events. Run: poetry run python scripts/build_interim.py")

    df_alerts = pd.read_parquet(alerts_path)
    df_events = pd.read_parquet(events_path)

    df_enriched = enrich_alerts_with_evidence(df_alerts, df_events, window_seconds=5, sample_events=10)

    out_parquet = Path("data/processed/alerts_hybrid_enriched_5s.parquet")
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_enriched.to_parquet(out_parquet, index=False)

    out_csv = export_alerts_csv(df_enriched, "data/processed/alerts_hybrid_enriched_5s.csv")

    print(f"Wrote {out_parquet} (rows={len(df_enriched)})")
    print(f"Wrote {out_csv} (rows={len(df_enriched)})")
    if len(df_enriched) > 0:
        cols = ["hybrid_score","severity","iforest_score","client_ip","server_ip","client_port","window_start","reasons","e_top_function_codes","e_top_addresses"]
        present = [c for c in cols if c in df_enriched.columns]
        print(df_enriched[present].head(10))


if __name__ == "__main__":
    main()
