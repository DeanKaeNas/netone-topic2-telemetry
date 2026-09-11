import json
import time
import random
import math
import zoneinfo
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import streamlit.components.v1 as components
import paho.mqtt.client as mqtt
import plotly.graph_objects as go

# Compatibility handling for TFLite across different deployment platforms
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="DIAO — NetOne Intelligence", layout="wide", initial_sidebar_state="collapsed")

BROKER = "broker.hivemq.com"
TOPIC = "diao/netone/telemetry"
SECTORS = list(range(0, 181, 15))
SECTOR_NAMES = {
    0: "Borrowdale", 15: "Highlands", 30: "Avondale", 45: "Mt Pleasant",
    60: "Greendale", 75: "CBD North", 90: "CBD Centre", 105: "CBD South",
    120: "Mbare", 135: "Highfields", 150: "Glen Norah", 165: "Budiriro", 180: "Chitungwiza"
}
# Physical node bearings (per hardware layout: Node A @ Borrowdale 30°, Node B @ Highfields 150°)
NODE_A_BEARING = 30
NODE_B_BEARING = 150

# ─────────────────────────────────────────────────────────────
# NOC DARK-MODE CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; font-family: 'Inter', 'Segoe UI', sans-serif; }
    h1 { color: #ffffff; font-weight: 700; letter-spacing: -0.5px; border-bottom: 1px solid #30363d; padding-bottom: 10px; }
    [data-testid="stMetricValue"] { color: #58a6ff !important; font-size: 2.3rem !important; font-family: 'Fira Code', monospace !important; }
    [data-testid="stMetricLabel"] { color: #8b949e !important; text-transform: uppercase; font-size: 0.82rem !important; letter-spacing: 0.5px; }
    [data-testid="stMetricDelta"] { background-color: #1f2428; padding: 2px 6px; border-radius: 4px; font-size: 0.8rem !important; }
    .status-live { color: #3fb950; font-weight: bold; text-shadow: 0 0 6px rgba(63,185,80,0.5); }
    .status-sim { color: #d29922; font-weight: bold; }
    .panel-title { color: #8b949e; text-transform: uppercase; letter-spacing: 1px; font-size: 0.85rem; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DATA / MODEL LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_hfss_data():
    df = pd.read_csv("NetOne_farfield_base.csv")
    df.columns = [c.strip().lower() for c in df.columns]
    phi_col = next(c for c in df.columns if "phi" in c)
    theta_col = next(c for c in df.columns if "theta" in c)
    gain_col = next(c for c in df.columns if "gain" in c)
    df[gain_col] = df[gain_col].astype(float).clip(lower=1e-10)
    df["gdb"] = 10 * np.log10(df[gain_col])
    lookup = {(int(r[phi_col]), int(r[theta_col])): r["gdb"] for _, r in df.iterrows()}
    e_plane = df[df[phi_col] == 0].sort_values(theta_col)
    h_plane = df[df[phi_col] == 90].sort_values(theta_col)
    return lookup, e_plane[theta_col].tolist(), e_plane["gdb"].tolist(), h_plane["gdb"].tolist()

@st.cache_resource
def load_ml_models():
    rf = joblib.load("diao_spatial_rf.pkl")
    interpreter = tflite.Interpreter(model_path="diao_nn.tflite")
    interpreter.allocate_tensors()
    return rf, interpreter, interpreter.get_input_details()[0], interpreter.get_output_details()[0]

def run_tflite_inference(interpreter, input_details, output_details, X_data):
    interpreter.set_tensor(input_details["index"], np.array(X_data, dtype=np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details["index"])[0]

lookup_table, angles_e, e_data, h_data = load_hfss_data()
rf_model, tflite_interp, input_det, output_det = load_ml_models()

def get_antenna_gain(phi, theta):
    p_val, t_val = max(0, min(360, int(phi))), max(0, min(180, int(theta)))
    for dp in (0, 5, -5, 10, -10):
        for dt in (0, 5, -5, 10, -10):
            val = lookup_table.get((p_val + dp, t_val + dt))
            if val is not None:
                return val
    return -10.0

# ─────────────────────────────────────────────────────────────
# MQTT — LIVE TELEMETRY
# ─────────────────────────────────────────────────────────────
def on_mqtt_message(client, userdata, msg):
    try:
        st.session_state.telemetry = json.loads(msg.payload.decode("utf-8"))
    except Exception:
        pass

if "telemetry" not in st.session_state:
    st.session_state.telemetry = None
if "sim_rssi_a" not in st.session_state:
    st.session_state.sim_rssi_a = -72.0
if "sim_rssi_b" not in st.session_state:
    st.session_state.sim_rssi_b = -85.0

if "mqtt_client" not in st.session_state:
    try:
        if hasattr(mqtt, "CallbackAPIVersion"):
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id="diao_streamlit_sub")
        else:
            client = mqtt.Client(client_id="diao_streamlit_sub")
        client.on_message = on_mqtt_message
        client.connect(BROKER, 1883, 60)
        client.subscribe(TOPIC)
        client.loop_start()
        st.session_state.mqtt_client = client
    except Exception:
        st.session_state.mqtt_client = None

telemetry_data = st.session_state.telemetry
if telemetry_data and telemetry_data.get("rssi_a") is not None:
    rssi_a = float(telemetry_data["rssi_a"])
    rssi_b = float(telemetry_data["rssi_b"])
    current_angle = telemetry_data.get("current_angle", 90)
    is_live = True
else:
    st.session_state.sim_rssi_a = max(-100, min(-45, st.session_state.sim_rssi_a + random.gauss(0, 2)))
    st.session_state.sim_rssi_b = max(-100, min(-45, st.session_state.sim_rssi_b + random.gauss(0, 2)))
    rssi_a = st.session_state.sim_rssi_a
    rssi_b = st.session_state.sim_rssi_b
    current_angle = 90
    is_live = False

cat_tz = zoneinfo.ZoneInfo("Africa/Harare")
now_cat = datetime.now(cat_tz)
gain_a = get_antenna_gain(0, 60) + random.uniform(-1, 1)
gain_b = get_antenna_gain(90, 60) + random.uniform(-1, 1)

# ─────────────────────────────────────────────────────────────
# AI ENSEMBLE INFERENCE
# ─────────────────────────────────────────────────────────────
X_features = [[rssi_a, rssi_b, gain_a, gain_b, now_cat.hour, now_cat.weekday()]]
rf_probs = rf_model.predict_proba(X_features)[0]
rf_predicted = int(rf_model.predict(X_features)[0])
rf_confidence = float(rf_probs.max())

nn_probs = run_tflite_inference(tflite_interp, input_det, output_det, X_features)
nn_predicted = SECTORS[int(np.argmax(nn_probs))]
nn_confidence = float(np.max(nn_probs))

if rf_predicted == nn_predicted:
    ai_angle = rf_predicted
    ensemble_confidence = (rf_confidence + nn_confidence) / 2
else:
    ai_angle, ensemble_confidence = (nn_predicted, nn_confidence) if nn_confidence >= rf_confidence else (rf_predicted, rf_confidence)

nearest_sector = min(SECTORS, key=lambda s: abs(s - current_angle))
ai_sector_name = SECTOR_NAMES.get(min(SECTORS, key=lambda s: abs(s - ai_angle)), "—")

# ─────────────────────────────────────────────────────────────
# RADAR WIDGET (SVG + CSS, live sweep, sector wedges, node blips)
# ─────────────────────────────────────────────────────────────
def polar_xy(theta_deg, r, cx=210, cy=210):
    """theta 0° = left horizon, 180° = right horizon, sweeping across the top."""
    math_deg = 180 - theta_deg
    rad = math.radians(math_deg)
    return cx + r * math.cos(rad), cy - r * math.sin(rad)

def rssi_to_radius(rssi, r_max=185, r_min=25):
    """Stronger signal (closer to -40 dBm) plots nearer the centre."""
    frac = max(0.0, min(1.0, (rssi - (-100)) / (60)))  # -100..-40 -> 0..1
    return r_max - frac * (r_max - r_min)

def render_radar(current_angle, ai_angle, rssi_a, rssi_b, confidence, live):
    cx, cy = 210, 210
    R = 185
    sweep_color = "#3fb950" if live else "#d29922"

    # Range rings (dBm labelled, outer=weak/-100, inner=strong/-40)
    rings = ""
    for frac, label in [(1.0, "-100"), (0.75, "-85"), (0.5, "-70"), (0.25, "-55"), (0.13, "-40")]:
        r = R * frac
        rings += f'<path d="M {cx-r},{cy} A {r},{r} 0 0 1 {cx+r},{cy}" fill="none" stroke="#21262d" stroke-width="1"/>'
        lx, ly = cx - r - 2, cy - 4
        rings += f'<text x="{lx}" y="{ly}" fill="#6e7681" font-size="9" text-anchor="end" font-family="Fira Code, monospace">{label}</text>'

    # Degree ticks + sector labels every 15°
    ticks = ""
    for deg, name in SECTOR_NAMES.items():
        x1, y1 = polar_xy(deg, R, cx, cy)
        x2, y2 = polar_xy(deg, R - 10, cx, cy)
        ticks += f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#30363d" stroke-width="1"/>'
        lx, ly = polar_xy(deg, R + 14, cx, cy)
        ticks += f'<text x="{lx:.1f}" y="{ly:.1f}" fill="#484f58" font-size="8" text-anchor="middle">{name}</text>'

    # Active sector wedge (current serving sector), highlighted
    def wedge(center_deg, width_deg, color, opacity):
        a1, a2 = center_deg - width_deg / 2, center_deg + width_deg / 2
        x1, y1 = polar_xy(a1, R, cx, cy)
        x2, y2 = polar_xy(a2, R, cx, cy)
        return f'<path d="M {cx},{cy} L {x1:.1f},{y1:.1f} A {R},{R} 0 0 1 {x2:.1f},{y2:.1f} Z" fill="{color}" opacity="{opacity}"/>'

    wedges = wedge(nearest_sector, 15, "#58a6ff", 0.12) + wedge(min(SECTORS, key=lambda s: abs(s - ai_angle)), 15, "#d2a8ff", 0.10)

    # Needles
    ncx, ncy = polar_xy(current_angle, R - 6, cx, cy)
    acx, acy = polar_xy(ai_angle, R - 6, cx, cy)
    needle_current = f'<line x1="{cx}" y1="{cy}" x2="{ncx:.1f}" y2="{ncy:.1f}" stroke="#58a6ff" stroke-width="2.5" style="filter:drop-shadow(0 0 4px #58a6ff);"/>'
    needle_ai = f'<line x1="{cx}" y1="{cy}" x2="{acx:.1f}" y2="{acy:.1f}" stroke="#d2a8ff" stroke-width="2" stroke-dasharray="4,3" style="filter:drop-shadow(0 0 4px #d2a8ff);"/>'

    # Node blips
    ax, ay = polar_xy(NODE_A_BEARING, rssi_to_radius(rssi_a), cx, cy)
    bx, by = polar_xy(NODE_B_BEARING, rssi_to_radius(rssi_b), cx, cy)
    blip_a = f'''<circle cx="{ax:.1f}" cy="{ay:.1f}" r="6" fill="#3fb950" style="filter:drop-shadow(0 0 6px #3fb950);"><animate attributeName="r" values="5;8;5" dur="2s" repeatCount="indefinite"/></circle>
                 <text x="{ax:.1f}" y="{ay-10:.1f}" fill="#3fb950" font-size="9" text-anchor="middle" font-family="Fira Code, monospace">A {rssi_a:.0f}</text>'''
    blip_b = f'''<circle cx="{bx:.1f}" cy="{by:.1f}" r="6" fill="#ff7b72" style="filter:drop-shadow(0 0 6px #ff7b72);"><animate attributeName="r" values="5;8;5" dur="2.3s" repeatCount="indefinite"/></circle>
                 <text x="{bx:.1f}" y="{by-10:.1f}" fill="#ff7b72" font-size="9" text-anchor="middle" font-family="Fira Code, monospace">B {rssi_b:.0f}</text>'''

    # Baseline + dead-zone hatch (antenna only sweeps 0-180, lower half inactive)
    baseline = f'<line x1="{cx-R}" y1="{cy}" x2="{cx+R}" y2="{cy}" stroke="#30363d" stroke-width="1"/>'

    html = f"""
    <div style="background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:6px;">
    <svg width="100%" height="420" viewBox="0 0 420 250" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <clipPath id="halfclip"><rect x="0" y="0" width="420" height="212"/></clipPath>
        </defs>
        <g clip-path="url(#halfclip)">
            <g style="transform-origin:{cx}px {cy}px;animation:spin 4s linear infinite;">
                <path d="M {cx},{cy} L {cx-R},{cy} A {R},{R} 0 0 1 {cx+R},{cy} Z"
                      fill="url(#sweepgrad)" opacity="0.5"/>
            </g>
            <radialGradient id="sweepgrad">
                <stop offset="0%" stop-color="{sweep_color}" stop-opacity="0.35"/>
                <stop offset="100%" stop-color="{sweep_color}" stop-opacity="0"/>
            </radialGradient>
            {rings}
            {wedges}
            {ticks}
            {baseline}
            {needle_ai}
            {needle_current}
            {blip_a}
            {blip_b}
            <circle cx="{cx}" cy="{cy}" r="4" fill="#c9d1d9"/>
        </g>
        <text x="{cx}" y="235" fill="#8b949e" font-size="10" text-anchor="middle" font-family="Fira Code, monospace">
            LIVE {current_angle:.0f}° &#8212; AI TARGET {ai_angle}° ({confidence:.0%} conf)
        </text>
    </svg>
    </div>
    <style>
        @keyframes spin {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
    </style>
    """
    return html

# ─────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────
st.title("DIAO — Dynamic Intelligent Antenna Optimization")

status_class = "status-live" if is_live else "status-sim"
status_text = "LIVE (MQTT)" if is_live else "SIMULATION (hardware offline)"
st.markdown(
    f"**NetOne Band 3 · 1800 MHz** | Status: <span class='{status_class}'>&#9679; {status_text}</span>",
    unsafe_allow_html=True,
)
if is_live and telemetry_data:
    esp_status = "Connected" if telemetry_data.get("esp32_connected") else "Simulation Mode"
    st.caption(f"ESP32: {esp_status} · {telemetry_data.get('decision_reason', '')}")
st.markdown("---")

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Current Angle", f"{current_angle}°", SECTOR_NAMES.get(nearest_sector, "—"))
col2.metric("AI Target Angle", f"{ai_angle}°", f"{ensemble_confidence:.0%} confidence")
col3.metric("Node A RSSI", f"{rssi_a:.0f} dBm", "Borrowdale · 30°")
col4.metric("Node B RSSI", f"{rssi_b:.0f} dBm", "Highfields · 150°")
col5.metric("RF Core Conf.", f"{rf_confidence:.0%}")
col6.metric("NN Edge Conf.", f"{nn_confidence:.0%}")

st.markdown("<br>", unsafe_allow_html=True)

radar_col, pattern_col = st.columns([1, 1])

with radar_col:
    st.markdown('<p class="panel-title">Live Sector Radar</p>', unsafe_allow_html=True)
    components.html(
        render_radar(current_angle, ai_angle, rssi_a, rssi_b, ensemble_confidence, is_live),
        height=430,
    )

with pattern_col:
    st.markdown('<p class="panel-title">HFSS Radiation Pattern</p>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=e_data, theta=angles_e, name="E-Plane",
        line=dict(color='#58a6ff', width=2), fill='toself', fillcolor='rgba(88,166,255,0.1)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=h_data, theta=angles_e, name="H-Plane",
        line=dict(color='#ff7b72', width=2), fill='toself', fillcolor='rgba(255,123,114,0.1)'
    ))
    fig.update_layout(
        height=430,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8b949e', family='Inter, sans-serif'),
        polar=dict(
            radialaxis=dict(visible=True, showline=False, gridcolor='#30363d', tickfont=dict(color='#8b949e')),
            angularaxis=dict(gridcolor='#30363d', tickfont=dict(color='#c9d1d9')),
            bgcolor='#0d1117'
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=20, b=10, l=10, r=10),
    )
    st.plotly_chart(fig, use_container_width=True)

time.sleep(2)
st.rerun()