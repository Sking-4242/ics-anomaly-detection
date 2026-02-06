from __future__ import annotations

from pathlib import Path

import pandas as pd

from icsad.protocols.modbus import decode_modbus_events_from_packets


def main() -> None:
    packets_path = Path("data/interim/packets.parquet")
    if not packets_path.exists():
        raise SystemExit("Missing data/interim/packets.parquet. Run: icsad parse <pcap> --out data/interim/packets.parquet")

    df_packets = pd.read_parquet(packets_path)

    df_modbus = decode_modbus_events_from_packets(df_packets)
    out_path = Path("data/interim/modbus_events.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_modbus.to_parquet(out_path, index=False)

    print(f"Wrote {out_path} (rows={len(df_modbus)})")
    if len(df_modbus) > 0:
        print(df_modbus.head(10))


if __name__ == "__main__":
    main()
