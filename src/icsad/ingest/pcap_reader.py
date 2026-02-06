from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class PacketRecord:
    ts: float
    src_ip: str
    dst_ip: str
    src_port: int | None
    dst_port: int | None
    ip_proto: int | None
    length: int
    payload_hex: str  # transport payload (TCP/UDP payload) as hex string


def _safe_int(x) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def read_pcap_packets(pcap_path: str | Path) -> list[PacketRecord]:
    """
    Read a pcap/pcapng and emit normalized packet records.
    We intentionally keep this layer protocol-agnostic.
    """
    pcap_path = Path(pcap_path)
    if not pcap_path.exists():
        raise FileNotFoundError(str(pcap_path))

    # Scapy import here so module import doesn't fail if scapy isn't installed yet
    from scapy.all import IP, IPv6, TCP, UDP, PcapReader  # type: ignore

    records: list[PacketRecord] = []
    with PcapReader(str(pcap_path)) as reader:
        for pkt in reader:
            ts = float(getattr(pkt, "time", 0.0))

            src_ip = ""
            dst_ip = ""
            ip_proto: int | None = None
            src_port: int | None = None
            dst_port: int | None = None
            payload_bytes = b""

            if IP in pkt:
                ip = pkt[IP]
                src_ip = getattr(ip, "src", "")
                dst_ip = getattr(ip, "dst", "")
                ip_proto = _safe_int(getattr(ip, "proto", None))
            elif IPv6 in pkt:
                ip6 = pkt[IPv6]
                src_ip = getattr(ip6, "src", "")
                dst_ip = getattr(ip6, "dst", "")
                ip_proto = _safe_int(getattr(ip6, "nh", None))

            if TCP in pkt:
                tcp = pkt[TCP]
                src_port = _safe_int(getattr(tcp, "sport", None))
                dst_port = _safe_int(getattr(tcp, "dport", None))
                # TCP payload only (not including headers)
                payload_bytes = bytes(tcp.payload) if tcp.payload is not None else b""
            elif UDP in pkt:
                udp = pkt[UDP]
                src_port = _safe_int(getattr(udp, "sport", None))
                dst_port = _safe_int(getattr(udp, "dport", None))
                payload_bytes = bytes(udp.payload) if udp.payload is not None else b""

            # Skip packets with no IP layer (ARP, etc.) for now
            if not src_ip or not dst_ip:
                continue

            records.append(
                PacketRecord(
                    ts=ts,
                    src_ip=src_ip,
                    dst_ip=dst_ip,
                    src_port=src_port,
                    dst_port=dst_port,
                    ip_proto=ip_proto,
                    length=len(pkt),
                    payload_hex=payload_bytes.hex(),
                )
            )

    return records


def to_dataframe(records: Iterable[PacketRecord]) -> pd.DataFrame:
    rows = [asdict(r) for r in records]
    df = pd.DataFrame(rows)
    if df.empty:
        # Ensure expected columns exist even if empty
        df = pd.DataFrame(
            columns=["ts", "src_ip", "dst_ip", "src_port", "dst_port", "ip_proto", "length", "payload_hex"]
        )
    return df


def write_parquet(df: pd.DataFrame, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


def pcap_to_parquet(pcap_path: str | Path, out_path: str | Path) -> Path:
    records = read_pcap_packets(pcap_path)
    df = to_dataframe(records)
    return write_parquet(df, out_path)



def parse_pcap_to_packets(pcap_path: str | Path) -> pd.DataFrame:
    """
    Convenience wrapper used by pipelines/UI.

    Returns a normalized packet DataFrame with columns:
    ts, src_ip, dst_ip, src_port, dst_port, ip_proto, length, payload_hex
    """
    records = read_pcap_packets(pcap_path)
    return to_dataframe(records)
