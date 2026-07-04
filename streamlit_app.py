#!/usr/bin/env python3
"""Streamlit web dashboard for the ML-based Intrusion Detection System."""

import os
import sys
import threading
import time
import argparse
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import IntrusionDetector, CONFIG

st.set_page_config(page_title="IDS Dashboard", layout="wide", page_icon=":shield:")


def parse_args():
    parser = argparse.ArgumentParser(description='IDS Streamlit Dashboard')
    parser.add_argument('--interface', type=str, default=None)
    parser.add_argument('--threshold', type=float, default=0.7)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--port', type=int, default=8501)
    return parser.parse_known_args()[0]


args = parse_args()
CONFIG['interface'] = args.interface
CONFIG['anomaly_threshold'] = args.threshold
CONFIG['training_mode'] = args.train


if 'detector' not in st.session_state:
    detector = IntrusionDetector(gui_mode=True)
    st.session_state.detector = detector
    thread = threading.Thread(target=detector.start, daemon=True)
    thread.start()
    st.session_state.thread = thread

detector = st.session_state.detector


st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; }
        .st-emotion-cache-1wivap2 { gap: 1rem; }
    </style>
""", unsafe_allow_html=True)

st.title(":shield: Intrusion Detection System")

status_col, mode_col = st.columns([1, 3])
with status_col:
    if CONFIG['training_mode']:
        st.info("**:material/sync: Training Mode**")
    elif detector.running:
        st.success("**:material/check_circle: Monitoring**")
    else:
        st.warning("**:material/pause_circle: Stopped**")

s = detector.stats
total = s['total_packets']
alert_rate = round(s['alerts'] / max(1, total) * 100, 2)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Packets", total)
m2.metric("Normal", s['normal'])
m3.metric("Alerts", s['alerts'], delta_color="inverse")
m4.metric("Alert Rate", f"{alert_rate}%")

alerts = list(detector.recent_alerts)
if alerts:
    st.markdown(f"### :warning: Recent Alerts ({len(alerts)})")
    cols = st.columns(min(3, len(alerts)))
    for i, a in enumerate(alerts[:6]):
        with cols[i % 3]:
            st.error(f"**{a.get('timestamp', '-')}**  \n"
                     f"{a.get('src', '?')} → {a.get('dst', '?')}  \n"
                     f"Score: {a.get('anomaly_score', 0):.4f}")

packets = list(detector.packet_buffer)
if packets:
    df = pd.DataFrame(packets)
    df['Status'] = df['is_alert'].apply(lambda x: ':warning:' if x else ':white_check_mark:')
    df['Flags'] = df['flags'].astype(str)
    df['Payload'] = df['payload_preview'].apply(lambda x: (str(x)[:40] + '..') if len(str(x)) > 40 else x)
    df['Score'] = df['anomaly_score'].apply(lambda x: f"{x:.4f}")
    display = df[['timestamp', 'src', 'dst', 'proto', 'length', 'Flags', 'Score', 'Status']].copy()
    display.columns = ['Time', 'Source', 'Destination', 'Proto', 'Len', 'Flags', 'Score', '']
    st.markdown("### Live Packets")
    st.dataframe(display.iloc[::-1].head(30), use_container_width=True, height=400, hide_index=True)

    with st.expander("Payload Previews"):
        for p in reversed(packets[-20:]):
            preview = p.get('payload_preview', '') or '(empty)'
            flag_str = f" [{p.get('flags', '-')}]" if p.get('flags') else ''
            st.code(f"[{p['timestamp']}] {p['src']} → {p['dst']}{flag_str}\n{preview[:200]}", language="text")
else:
    st.info("Waiting for packets...")

if packets and not CONFIG['training_mode']:
    st.markdown("### Traffic Charts")
    c1, c2 = st.columns(2)
    with c1:
        chart_df = pd.DataFrame({
            'idx': range(len(packets)),
            'Packet Length': [p['length'] for p in packets],
            'Anomaly Score': [p.get('anomaly_score', 0) for p in packets]
        })
        st.line_chart(chart_df.set_index('idx'))
    with c2:
        proto_counts = df['proto'].value_counts()
        st.bar_chart(proto_counts)

time.sleep(2)
st.rerun()
