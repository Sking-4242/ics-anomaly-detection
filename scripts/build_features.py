from __future__ import annotations

from pathlib import Path

import pandas as pd

from icsad.features.base import WindowSpec
from icsad.features.packet_features import build_modbus_packet_features
from icsad.features.session_features import build_modbus_window_features


def main() -> None:
    modbus_path = Path("data/interim/modbus_events.parquet")
    if not modbus_path.exists():
        raise SystemExit("Missing data/interim/modbus_events.parquet. Run: poetry run python scripts/build_interim.py")

    df_modbus = pd.read_parquet(modbus_path)

    # Packet features
    df_pkt = build_modbus_packet_features(df_modbus)
    out_pkt = Path("data/processed/modbus_packet_features.parquet")
    out_pkt.parent.mkdir(parents=True, exist_ok=True)
    df_pkt.to_parquet(out_pkt, index=False)

    # Window features (5 seconds locked)
    window = WindowSpec(seconds=5)
    df_win = build_modbus_window_features(df_modbus, window=window)
    out_win = Path("data/processed/modbus_window_features_5s.parquet")
    df_win.to_parquet(out_win, index=False)

    print(f"Wrote {out_pkt} (rows={len(df_pkt)})")
    print(f"Wrote {out_win} (rows={len(df_win)})")
    print(df_win.head(10))


if __name__ == "__main__":
    main()
