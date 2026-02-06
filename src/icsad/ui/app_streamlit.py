from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import streamlit as st


HYBRID_PATH = Path("data/processed/alerts_hybrid_enriched_5s.parquet")
BASELINE_PATH = Path("data/processed/alerts_enriched_5s.parquet")
IF_EVAL_PATH = Path("data/processed/eval_iforest_top50_5s.csv")


WRITE_FCS = {5, 6, 15, 16}  # single/multiple coil/register writes in Modbus


def _safe_json(obj) -> str:
    try:
        return json.dumps(obj, indent=2)
    except Exception:
        return str(obj)


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)

    # Defensive: parquet may contain duplicate column names; Streamlit/PyArrow will crash otherwise.
    df = df.loc[:, ~df.columns.duplicated()].copy()

    if "reasons" in df.columns:
        df["reasons_text"] = df["reasons"].apply(lambda x: " | ".join(x) if isinstance(x, list) else str(x))
    else:
        df["reasons_text"] = ""

    # Normalize a couple columns if present
    if "dt_min" in df.columns:
        df["dt_min"] = pd.to_numeric(df["dt_min"], errors="coerce")
    if "event_count" in df.columns:
        df["event_count"] = pd.to_numeric(df["event_count"], errors="coerce")

    return df


def _kpis(df: pd.DataFrame) -> dict[str, str]:
    if df.empty:
        return {"alerts": "0", "unique_flows": "0", "top_server": "-", "median_dt_min": "-"}

    flow_cols = [c for c in ["client_ip", "server_ip", "client_port", "server_port"] if c in df.columns]
    unique_flows = df[flow_cols].dropna().drop_duplicates().shape[0] if flow_cols else 0

    top_server = "-"
    if "server_ip" in df.columns and df["server_ip"].notna().any():
        top_server = df["server_ip"].value_counts().index[0]

    median_dt_min = "-"
    if "dt_min" in df.columns:
        v = pd.to_numeric(df["dt_min"], errors="coerce").median()
        if pd.notna(v):
            median_dt_min = f"{float(v):.6f}"

    return {
        "alerts": str(len(df)),
        "unique_flows": str(unique_flows),
        "top_server": str(top_server),
        "median_dt_min": str(median_dt_min),
    }


def _filter_df(df: pd.DataFrame, server_sel: str, client_sel: str, score_col: str | None, score_range: tuple[float, float]) -> pd.DataFrame:
    dff = df.copy()
    if server_sel != "(all)" and "server_ip" in dff.columns:
        dff = dff[dff["server_ip"] == server_sel]
    if client_sel != "(all)" and "client_ip" in dff.columns:
        dff = dff[dff["client_ip"] == client_sel]
    if score_col and score_col in dff.columns:
        dff = dff[(dff[score_col] >= score_range[0]) & (dff[score_col] <= score_range[1])]
    return dff


def _dedupe_list(seq: list[str]) -> list[str]:
    # preserve order
    return list(dict.fromkeys(seq))


def _alerts_table(df: pd.DataFrame, score_col: str) -> None:
    show_cols = [c for c in [
        score_col, "severity", "iforest_score", "write_risk",
        "client_ip", "server_ip", "client_port", "server_port",
        "window_start", "event_count", "request_count", "response_count",
        "dt_min",
        "reasons_text",
    ] if c in df.columns]
    show_cols = _dedupe_list(show_cols)
    if "write_risk" in show_cols:
        df = df.copy()
        df["write_risk"] = df.apply(_write_badge, axis=1)
    st.dataframe(df[show_cols].reset_index(drop=True), width="stretch", height=360)


def _write_badge(row: dict) -> str:
    return "WRITE" if _has_write_activity(row) else "READ"

def _has_write_activity(row: dict) -> bool:
    # Check top function codes list
    tfc = row.get("e_top_function_codes")
    if isinstance(tfc, list):
        for item in tfc:
            if isinstance(item, dict):
                v = item.get("value")
                try:
                    if int(v) in WRITE_FCS:
                        return True
                except Exception:
                    pass

    # Fallback: check sample events
    samp = row.get("e_sample_events")
    if isinstance(samp, list):
        for ev in samp:
            if isinstance(ev, dict):
                try:
                    if int(ev.get("fc", -1)) in WRITE_FCS:
                        return True
                except Exception:
                    pass

    return False


def _timing_risk(row: dict) -> str:
    """
    Translate dt_min into an easy-to-read risk bucket.
    """
    dt = row.get("dt_min")
    try:
        dt = float(dt)
    except Exception:
        return "Unknown"

    if dt <= 0:
        return "Unknown"
    if dt <= 0.00010:
        return "High (very tight bursts)"
    if dt <= 0.00020:
        return "Medium (bursty timing)"
    if dt <= 0.001:
        return "Low-Med (fast polling)"
    return "Low (normal-ish timing)"


def _risk_indicator(row: dict) -> tuple[str, str]:
    """
    Returns (label, detail).
    """
    write = _has_write_activity(row)
    timing = _timing_risk(row)

    if write:
        return ("HIGH RISK: Write behavior present", f"Writes (fc in {sorted(WRITE_FCS)}) can change process state. Timing: {timing}")
    # No writes; timing still matters
    if timing.startswith("High"):
        return ("MED-HIGH RISK: Extreme timing burst", f"No writes observed, but dt_min suggests automation/replay. Timing: {timing}")
    if timing.startswith("Medium"):
        return ("MEDIUM RISK: Bursty timing", f"No writes observed; timing indicates bursty activity. Timing: {timing}")
    return ("LOWER RISK: Read/normal-like behavior", f"No writes observed; timing appears less bursty. Timing: {timing}")


def _glossary_sidebar() -> None:
    st.sidebar.header("Glossary")
    st.sidebar.markdown(
        """
**ICS / OT:** Industrial Control Systems / Operational Technology; systems that monitor and control physical processes.

**PLC:** Programmable Logic Controller; an industrial computer that executes control logic.

**SCADA / HMI:** Supervisory control systems and operator interfaces used to monitor and control industrial processes.

**Modbus TCP (Port 502):** A widely used industrial protocol for client/server communication with PLCs.

**Register / Coil / Address:** Logical memory locations representing process values, states, or outputs.

**Function code (fc):** Specifies the Modbus operation being performed:
- **Reads:** fc 1/2/3/4 (lower risk)
- **Writes:** fc **5/6/15/16** (higher risk: changes state)

**5-second window:** A fixed time interval used to aggregate events for behavioral analysis.

**dt_min:** Minimum inter-event time within the window; extremely small values indicate bursty or automated behavior.

**Exception:** Protocol-level error response indicating invalid requests, faults, or probing.
        """
    )


def _help_panel() -> None:
    with st.expander("Explain this dashboard (for non-ICS readers)", expanded=True):
        st.markdown(
            """
### What you’re looking at
This dashboard summarizes **industrial control system (ICS) network traffic** captured in a PCAP (packet capture).
It focuses on **Modbus TCP**, a common protocol used for PLC/RTU communications in manufacturing, water, energy, and building automation.

### Why this matters
In ICS environments, abnormal network behavior can indicate:
- **misconfiguration** (devices talking incorrectly),
- **equipment failure** (timeouts, retries, error responses),
- **unauthorized access** (scanning, replay, scripted writes), or
- **malicious activity** (attempts to change physical process behavior).

Unlike office IT, ICS incidents can affect **availability and physical safety**, not just data.

---

## Key concepts
### 1) Windows (5-second bins)
We convert raw packets into protocol events and then group them into **5-second windows** per client↔server flow.
Each row corresponds to one flow during one window.

### 2) Client vs Server
- **Client**: typically the HMI / engineering workstation / SCADA master.
- **Server**: typically the PLC/RTU (Modbus TCP listens on **port 502**).

---

## Detection methods in this tool
### Baseline (rules)
Human-readable rules flag suspicious patterns (fast bursts, imbalance, exceptions, unusual diversity).

### Isolation Forest (unsupervised ML)
Scores windows that look statistically unusual compared to the dataset.

### Hybrid score (recommended)
Combines baseline explainability (the “why”) with ML ranking (the “how unusual”) and enforces a realistic **alert budget** (top-K).

---

## How to interpret the alert details
### Risk indicator
- **Write behavior present = higher risk** (writes can change equipment state).
- Tight timing bursts can indicate automation/replay, even if only reads are observed.

### Evidence fields
- **e_top_function_codes**: most common operations in this window.
- **e_top_addresses**: which register/coil addresses were accessed.
- **e_top_quantities**: how many were read/written at a time.
- **e_exception_codes**: device error responses.
- **e_sample_events**: timestamped excerpt of events for quick validation.
            """
        )


def _alert_details(df: pd.DataFrame, score_col: str) -> None:
    if df.empty:
        st.info("No rows after filtering.")
        return

    idx = st.number_input("Select alert row index", min_value=0, max_value=max(0, len(df) - 1), value=0, step=1)
    row = df.reset_index(drop=True).iloc[int(idx)].to_dict()

    risk_label, risk_detail = _risk_indicator(row)

    # Top risk banner
    if risk_label.startswith("HIGH"):
        st.error(risk_label)
    elif "MED" in risk_label:
        st.warning(risk_label)
    else:
        st.info(risk_label)

    st.caption(risk_detail)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Score", f"{float(row.get(score_col, 0.0)):.4f}")
    with c2:
        st.metric("Baseline severity", f"{float(row.get('severity', 0.0)):.4f}")
    with c3:
        st.metric("IF score", f"{float(row.get('iforest_score', 0.0)):.4f}")
    with c4:
        st.metric("Timing risk", _timing_risk(row))

    st.write("Flow")
    st.code(
        f"{row.get('client_ip')}:{row.get('client_port')} -> {row.get('server_ip')}:{row.get('server_port')}\n"
        f"window_start={row.get('window_start')}"
    )

    st.write("Reasons")
    st.info(row.get("reasons_text", ""))

    ev_cols = ["e_top_function_codes", "e_top_addresses", "e_top_quantities", "e_exception_codes", "e_sample_events"]
    for col in ev_cols:
        if col in row:
            st.write(col)
            st.code(_safe_json(row[col]))


def _flow_timeline(df: pd.DataFrame) -> None:
    if df.empty or "window_start" not in df.columns:
        return

    st.subheader("Flow timeline (alerts per 5s window)")

    flows = df[["client_ip", "server_ip", "client_port", "server_port"]].dropna().drop_duplicates()
    flows["flow_str"] = flows.apply(
        lambda r: f"{r['client_ip']}:{r['client_port']} -> {r['server_ip']}:{r['server_port']}", axis=1
    )
    flow_list = flows["flow_str"].tolist()
    if not flow_list:
        st.info("No flow information available.")
        return

    sel = st.selectbox("Select flow", flow_list)
    chosen = flows[flows["flow_str"] == sel].iloc[0].to_dict()

    dff = df[
        (df["client_ip"] == chosen["client_ip"])
        & (df["server_ip"] == chosen["server_ip"])
        & (df["client_port"] == chosen["client_port"])
        & (df["server_port"] == chosen["server_port"])
    ].copy()

    dff["window_start"] = pd.to_numeric(dff["window_start"], errors="coerce").fillna(0.0)
    counts = dff.groupby("window_start").size().reset_index(name="alert_count").sort_values("window_start")
    st.dataframe(counts, width="stretch", height=220)


def main() -> None:
    st.set_page_config(page_title="ICSAD Dashboard", layout="wide")
    st.title("ICSAD: Analyst Dashboard (Timing-first, Hybrid)")

    # Sidebar glossary
    _glossary_sidebar()

    # In-page help panel
    _help_panel()

    tab_h, tab_b, tab_if = st.tabs(["Hybrid (Enriched)", "Baseline (Enriched)", "Isolation Forest (Top-50)"])

    # ---------------- Hybrid Tab ----------------
    with tab_h:
        df = _load_parquet(HYBRID_PATH)
        if df.empty:
            st.warning("Hybrid enriched alerts not found. Run: poetry run python scripts/run_hybrid.py && poetry run python scripts/enrich_hybrid_alerts.py")
            st.stop()

        k = _kpis(df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Alerts", k["alerts"])
        c2.metric("Unique flows", k["unique_flows"])
        c3.metric("Top server", k["top_server"])
        c4.metric("Median dt_min", k["median_dt_min"])

        st.sidebar.header("Filters (Hybrid/Baseline)")
        servers = sorted(df["server_ip"].dropna().unique().tolist()) if "server_ip" in df.columns else []
        clients = sorted(df["client_ip"].dropna().unique().tolist()) if "client_ip" in df.columns else []
        server_sel = st.sidebar.selectbox("Server IP", ["(all)"] + servers, key="hy_server")
        client_sel = st.sidebar.selectbox("Client IP", ["(all)"] + clients, key="hy_client")

        score_col = "hybrid_score" if "hybrid_score" in df.columns else None
        if score_col:
            mn = float(pd.to_numeric(df[score_col], errors="coerce").min())
            mx = float(pd.to_numeric(df[score_col], errors="coerce").max())
            score_range = st.sidebar.slider("Hybrid score range", min_value=mn, max_value=mx, value=(mn, mx), key="hy_score")
        else:
            score_range = (0.0, 0.0)

        dff = _filter_df(df, server_sel, client_sel, score_col, score_range)
        sort_col = st.selectbox("Sort by", [c for c in ["hybrid_score", "severity", "iforest_score", "window_start"] if c in dff.columns], index=0)
        dff = dff.sort_values(sort_col, ascending=False)

        st.subheader("Alerts")
        _alerts_table(dff, "hybrid_score")
        st.download_button("Download filtered CSV", dff.to_csv(index=False).encode("utf-8"), "hybrid_filtered.csv", "text/csv")

        st.divider()
        st.subheader("Alert details")
        _alert_details(dff, "hybrid_score")

        st.divider()
        _flow_timeline(dff)

    # ---------------- Baseline Tab ----------------
    with tab_b:
        dfb = _load_parquet(BASELINE_PATH)
        if dfb.empty:
            st.warning("Baseline enriched alerts not found. Run: poetry run python scripts/enrich_alerts.py")
        else:
            kb = _kpis(dfb)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Alerts", kb["alerts"])
            c2.metric("Unique flows", kb["unique_flows"])
            c3.metric("Top server", kb["top_server"])
            c4.metric("Median dt_min", kb["median_dt_min"])

            servers = sorted(dfb["server_ip"].dropna().unique().tolist()) if "server_ip" in dfb.columns else []
            clients = sorted(dfb["client_ip"].dropna().unique().tolist()) if "client_ip" in dfb.columns else []
            server_sel = st.selectbox("Server IP", ["(all)"] + servers, key="bl_server")
            client_sel = st.selectbox("Client IP", ["(all)"] + clients, key="bl_client")

            dffb = _filter_df(dfb, server_sel, client_sel, None, (0.0, 0.0))
            sort_col = st.selectbox("Sort by", [c for c in ["severity", "window_start"] if c in dffb.columns], index=0)
            dffb = dffb.sort_values(sort_col, ascending=False)

            st.subheader("Alerts")
            _alerts_table(dffb, "severity")
            st.download_button("Download filtered CSV", dffb.to_csv(index=False).encode("utf-8"), "baseline_filtered.csv", "text/csv")

            st.divider()
            st.subheader("Alert details")
            _alert_details(dffb, "severity")

            st.divider()
            _flow_timeline(dffb)

    # ---------------- IF Tab ----------------
    with tab_if:
        if not IF_EVAL_PATH.exists():
            st.warning("IF evaluation CSV not found. Run: poetry run python scripts/run_iforest.py")
        else:
            dfi = pd.read_csv(IF_EVAL_PATH)
            st.subheader("Isolation Forest (Top-50) ranked windows")
            st.dataframe(dfi.head(50), width="stretch", height=420)
            st.download_button("Download IF eval CSV", IF_EVAL_PATH.read_bytes(), "eval_iforest_top50_5s.csv", "text/csv")


if __name__ == "__main__":
    main()
