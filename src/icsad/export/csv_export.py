from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_alerts_csv(df_alerts: pd.DataFrame, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = df_alerts.copy()
    # reasons is a list -> make it readable in CSV
    if "reasons" in df.columns:
        df["reasons"] = df["reasons"].apply(lambda x: " | ".join(x) if isinstance(x, list) else str(x))

    df.to_csv(out_path, index=False)
    return out_path
