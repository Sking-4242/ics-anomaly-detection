from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import pandas as pd

from icsad.ingest.pcap_reader import parse_pcap_to_packets
from icsad.protocols.modbus import decode_modbus_events_from_packets
from icsad.features.packet_features import build_modbus_packet_features
from icsad.features.session_features import build_modbus_window_features
from icsad.features.base import WindowSpec
from icsad.detection.iforest import IFConfig, train_iforest, score_iforest
from icsad.detection.hybrid import build_hybrid_alerts
from icsad.explain.report import enrich_alerts_with_evidence


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", required=True, help="Path to .pcap/.pcapng")
    ap.add_argument("--outdir", required=True, help="Output directory for artifacts")
    ap.add_argument("--window", type=int, default=5, help="Window size in seconds")
    ap.add_argument("--topk", type=int, default=50, help="Top-K alerts to keep")
    args = ap.parse_args()

    pcap = Path(args.pcap).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Copy pcap into outdir for traceability
    pcap_copy = outdir / pcap.name
    if pcap != pcap_copy:
        shutil.copy2(pcap, pcap_copy)

    # 1) Packets
    df_packets = parse_pcap_to_packets(str(pcap_copy))
    packets_path = outdir / "packets.parquet"
    df_packets.to_parquet(packets_path, index=False)

    # 2) Modbus events
    df_events = decode_modbus_events_from_packets(df_packets)
    events_path = outdir / "modbus_events.parquet"
    df_events.to_parquet(events_path, index=False)

    # 3) Features
    df_pkt_feat = build_modbus_packet_features(df_events)
    pkt_feat_path = outdir / "modbus_packet_features.parquet"
    df_pkt_feat.to_parquet(pkt_feat_path, index=False)

    df_win_feat = build_modbus_window_features(df_events, window=WindowSpec(seconds=args.window))
    win_feat_path = outdir / f"modbus_window_features_{args.window}s.parquet"
    df_win_feat.to_parquet(win_feat_path, index=False)

    # 4) IF scoring on windows
    cfg = IFConfig(contamination=0.05)
    model = train_iforest(df_win_feat, cfg)
    df_scored = score_iforest(model, df_win_feat)

    # 5) Baseline-ish alerts input: reuse your existing baseline severity if present,
    # but for now we construct a minimal baseline using dt_min bursts similar to your earlier rules.
    # If you want, we can import your exact baseline engine instead.
    df_base = df_win_feat.copy()
    # A simple baseline severity proxy: flag small dt_min
    # (keeps the hybrid functional even if baseline scripts change)
    df_base["severity"] = 0.0
    df_base.loc[df_base["dt_min"] <= 0.0002, "severity"] = 3.0
    df_base["reasons"] = df_base.apply(
        lambda r: ["DT_MIN: Very small inter-arrival time (burst)"] if float(r.get("dt_min", 1.0) or 1.0) <= 0.0002 else [],
        axis=1,
    )
    df_base = df_base[df_base["severity"] > 0].copy()

    # 6) Hybrid + gating
    df_hybrid = build_hybrid_alerts(df_base, df_scored, alpha=0.6, beta=0.4, top_k=args.topk)

    # 7) Enrich with evidence (uses raw modbus events)
    df_enriched = enrich_alerts_with_evidence(df_hybrid, df_events, window_seconds=args.window, sample_events=10)

    out_alerts = outdir / f"alerts_hybrid_enriched_{args.window}s.parquet"
    df_enriched.to_parquet(out_alerts, index=False)
    df_enriched.to_csv(outdir / f"alerts_hybrid_enriched_{args.window}s.csv", index=False)

    print(f"Wrote: {out_alerts} (rows={len(df_enriched)})")


if __name__ == "__main__":
    main()
