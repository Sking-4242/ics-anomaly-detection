from pathlib import Path

import pandas as pd

from icsad.ingest.pcap_reader import pcap_to_parquet


def test_pcap_to_parquet_smoke(tmp_path: Path):
    sample_pcap = Path("data/raw/sample.pcap")
    if not sample_pcap.exists():
        # Local dev convenience: skip if user hasn't added a sample yet
        return

    out = tmp_path / "packets.parquet"
    written = pcap_to_parquet(sample_pcap, out)

    assert written.exists()
    df = pd.read_parquet(written)
    expected_cols = {"ts", "src_ip", "dst_ip", "src_port", "dst_port", "ip_proto", "length", "payload_hex"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) > 0
