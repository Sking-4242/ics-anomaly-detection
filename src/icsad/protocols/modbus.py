from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import pandas as pd

MODBUS_TCP_PORT = 502


@dataclass(frozen=True)
class ModbusEvent:
    ts: float
    src_ip: str
    dst_ip: str
    src_port: int | None
    dst_port: int | None

    # MBAP
    transaction_id: int
    protocol_id: int
    length: int  # length field in MBAP (unit id + PDU)
    unit_id: int

    # PDU
    function_code: int
    is_exception: bool
    exception_code: int | None

    # Semantics (best-effort)
    direction: str  # "request" | "response" | "unknown"
    address: int | None
    quantity: int | None
    value: int | None  # for single write, etc. (best-effort)

    raw_len: int  # bytes of TCP payload


def _u16(b: bytes, off: int) -> int:
    return (b[off] << 8) | b[off + 1]


def _parse_mbap(payload: bytes) -> Optional[tuple[int, int, int, int]]:
    # MBAP header is 7 bytes: TID(2), PID(2), LEN(2), UID(1)
    if len(payload) < 7:
        return None
    tid = _u16(payload, 0)
    pid = _u16(payload, 2)
    length = _u16(payload, 4)
    uid = payload[6]
    return tid, pid, length, uid


def _parse_pdu(payload: bytes) -> Optional[tuple[int, bool, Optional[int], bytes]]:
    # PDU begins immediately after MBAP (7 bytes)
    if len(payload) < 8:
        return None
    fc = payload[7]
    is_exc = (fc & 0x80) != 0
    if is_exc:
        exc_code = payload[8] if len(payload) >= 9 else None
        return fc, True, exc_code, payload[8:]
    return fc, False, None, payload[8:]


def _infer_direction(src_port: int | None, dst_port: int | None, pdu_rest: bytes) -> str:
    # Modbus TCP server typically listens on 502.
    # If dst_port=502 -> request. If src_port=502 -> response.
    if dst_port == MODBUS_TCP_PORT:
        return "request"
    if src_port == MODBUS_TCP_PORT:
        return "response"
    # fallback heuristic: requests often have fixed-length small PDUs for reads/writes
    if len(pdu_rest) in (4, 5):  # common request body sizes
        return "request"
    return "unknown"


def _extract_semantics(fc: int, pdu_rest: bytes, direction: str) -> tuple[int | None, int | None, int | None]:
    """
    Best-effort semantic extraction for common function codes.
    For requests, pdu_rest often begins with address(2) + quantity/value(2) ...
    For responses, shapes vary; we keep semantics mostly for requests initially.
    """
    address = quantity = value = None

    # Only attempt address/quantity on likely requests
    if direction != "request":
        return address, quantity, value

    # Common read functions: 0x01/0x02/0x03/0x04 => address(2), quantity(2)
    if fc in (0x01, 0x02, 0x03, 0x04) and len(pdu_rest) >= 4:
        address = _u16(pdu_rest, 0)
        quantity = _u16(pdu_rest, 2)
        return address, quantity, value

    # Write single coil/register: 0x05/0x06 => address(2), value(2)
    if fc in (0x05, 0x06) and len(pdu_rest) >= 4:
        address = _u16(pdu_rest, 0)
        value = _u16(pdu_rest, 2)
        return address, quantity, value

    # Write multiple coils/registers: 0x0F/0x10 => address(2), quantity(2), bytecount(1), values...
    if fc in (0x0F, 0x10) and len(pdu_rest) >= 5:
        address = _u16(pdu_rest, 0)
        quantity = _u16(pdu_rest, 2)
        return address, quantity, value

    return address, quantity, value


def decode_modbus_events_from_packets(df_packets: pd.DataFrame) -> pd.DataFrame:
    """
    Input df_packets must contain:
    ts, src_ip, dst_ip, src_port, dst_port, payload_hex
    Returns a DataFrame of decoded Modbus TCP events for port-based Modbus flows.
    """
    events: list[ModbusEvent] = []

    required = {"ts", "src_ip", "dst_ip", "src_port", "dst_port", "payload_hex"}
    missing = required - set(df_packets.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Filter likely Modbus TCP by port 502
    df = df_packets[(df_packets["src_port"] == MODBUS_TCP_PORT) | (df_packets["dst_port"] == MODBUS_TCP_PORT)].copy()

    for row in df.itertuples(index=False):
        payload_hex = getattr(row, "payload_hex")
        if not isinstance(payload_hex, str) or len(payload_hex) < 16:
            continue

        try:
            payload = bytes.fromhex(payload_hex)
        except ValueError:
            continue

        mbap = _parse_mbap(payload)
        if mbap is None:
            continue
        tid, pid, mbap_len, uid = mbap

        pdu = _parse_pdu(payload)
        if pdu is None:
            continue
        fc, is_exc, exc_code, pdu_rest = pdu

        src_port = getattr(row, "src_port")
        dst_port = getattr(row, "dst_port")
        direction = _infer_direction(src_port, dst_port, pdu_rest)
        address, quantity, value = _extract_semantics(fc, pdu_rest, direction)

        events.append(
            ModbusEvent(
                ts=float(getattr(row, "ts")),
                src_ip=str(getattr(row, "src_ip")),
                dst_ip=str(getattr(row, "dst_ip")),
                src_port=int(src_port) if src_port is not None else None,
                dst_port=int(dst_port) if dst_port is not None else None,
                transaction_id=tid,
                protocol_id=pid,
                length=mbap_len,
                unit_id=uid,
                function_code=fc,
                is_exception=is_exc,
                exception_code=exc_code,
                direction=direction,
                address=address,
                quantity=quantity,
                value=value,
                raw_len=len(payload),
            )
        )

    out = pd.DataFrame([asdict(e) for e in events])
    if out.empty:
        out = pd.DataFrame(
            columns=[
                "ts",
                "src_ip",
                "dst_ip",
                "src_port",
                "dst_port",
                "transaction_id",
                "protocol_id",
                "length",
                "unit_id",
                "function_code",
                "is_exception",
                "exception_code",
                "direction",
                "address",
                "quantity",
                "value",
                "raw_len",
            ]
        )
    return out
