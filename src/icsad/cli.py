from __future__ import annotations

import argparse
from pathlib import Path

from icsad.ingest.pcap_reader import pcap_to_parquet


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="icsad", description="ICS/SCADA anomaly detection tooling")
    sub = p.add_subparsers(dest="cmd", required=True)

    parse = sub.add_parser("parse", help="Parse a PCAP into normalized packet/event records")
    parse.add_argument("pcap", help="Path to .pcap/.pcapng file")
    parse.add_argument("--out", default="data/interim/packets.parquet", help="Output Parquet path")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "parse":
        pcap_path = Path(args.pcap)
        out_path = Path(args.out)
        written = pcap_to_parquet(pcap_path, out_path)
        print(f"Wrote {written}")
        return

    raise SystemExit("Unknown command")


if __name__ == "__main__":
    main()
