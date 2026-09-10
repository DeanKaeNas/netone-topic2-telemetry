import json
import time
import random
import zoneinfo
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import paho.mqtt.client as mqtt
import plotly.graph_objects as go

# Compatibility handling for TFLite
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# --- Page Config (Must be first) ---
st.set_page_config(page_title="DIAO — NetOne Intelligence", layout="wide", initial_sidebar_state="collapsed")

# --- Custom CSS injected for the "NetworkOps" NOC Aesthetic ---
st.markdown("""
<style>
    /* Dark theme background with slight blue tint */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    /* Sleek header styling */
    h1 {
        color: #ffffff;
        font-weight: 700;
        letter-spacing: -0.5px;
        border-bottom: 1px solid #30363d;
        padding-bottom: 10px;
    }
    
    /* Style metric cards to look like hardware readouts */
    [data-testid="stMetricValue"] {
        color: #58a6ff !important;
        font-size: 2.5rem !important;
        font-family: 'Fira Code', monospace !important;
    }
    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
        text-transform: uppercase;
        font-size: 0.85rem !important;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetricDelta"] {
        background-color: #1f2428;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.8rem !important;
    }
    
    /* Neon Status Indicator */
    .status-live {
        color: #3fb950;
        font-weight: bold;
        text-shadow: 0 0 5px rgba(63, 185, 80, 0.4);
    }
    .status-sim {
        color: #d29922;
        font-weight: bold;
    }
    
    /* Clean up dataframe and container borders */
    .css-1v0mbdj > img {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

BROKER = "broker.hivemq.com"
TOPIC = "diao/netone/telemetry"
SECTORS = list(range(0, 181, 15))
SECTOR_NAMES = {
    0: "Borrowdale", 15: "Highlands", 30: "Avondale", 45: "Mt Pleasant",
    60: "Greendale", 75: "CBD North", 90: "CBD Centre", 105: "CBD South",
    120: "Mbare", 135: "Highfields", 150: "Glen Norah", 165: "Budiriro", 180: "Chitungwiza"
}

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
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    return rf, interpreter, input_details, output_details

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

# MQTT Telemetry Handler
def on_mqtt_message(client, userdata, msg):
    try:
        st.session_state.telemetry = json.loads(msg.payload.decode("utf-8"))
    except Exception:
        pass

# Initialize state
if "telemetry" not in st.session_state:
    st.session_state.telemetry = None
if "sim_rssi_a" not in st.session_state:
    st.session_state.sim_rssi_a = -72.0
if "sim_rssi_b" not in st.session_state:
    st.session_state.sim_rssi_b = -85.0

# Persistent MQTT connection
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
current_hour = now_cat.hour
current_dow = now_cat.weekday()

gain_a = get_antenna_gain(0, 60) + random.uniform(-1, 1)
gain_b = get_antenna_gain(90, 60) + random.uniform(-1, 1)

# AI Inference
X_features = [[rssi_a, rssi_b, gain_a, gain_b, current_hour, current_dow]]
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
    if nn_confidence >= rf_confidence:
        ai_angle = nn_predicted
        ensemble_confidence = nn_confidence
    else:
        ai_angle = rf_predicted
        ensemble_confidence = rf_confidence

# --- Dashboard Layout ---
st.title("DIAO — Dynamic Intelligent Antenna Optimization")

status_class = "status-live" if is_live else "status-sim"
status_text = "LIVE (MQTT gRPC Emulation)" if is_live else "SIMULATION (Hardware Disconnected)"
st.markdown(f"**NetOne Band 3 · 1800 MHz** | Status: <span class='{status_class}'>● {status_text}</span>", unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3, col4, col5, col6 = st.columns(6)
nearest_sector = min(SECTORS, key=lambda s: abs(s - current_angle))

# We use the 'delta' parameter to act as a secondary label below the main metric
col1.metric("Current Angle", f"{current_angle}°", SECTOR_NAMES.get(nearest_sector, "—"))
col2.metric("AI Target Angle", f"{ai_angle}°", f"{ensemble_confidence:.0%} confidence")
col3.metric("Node A RSSI", f"{rssi_a:.0f} dBm", "Primary Path")
col4.metric("Node B RSSI", f"{rssi_b:.0f} dBm", "Secondary Path")
col5.metric("RF Core Conf.", f"{rf_confidence:.0%}")
col6.metric("NN Edge Conf.", f"{nn_confidence:.0%}")

st.markdown("<br>", unsafe_allow_html=True)

# --- High-Tech Plotly Radar Chart ---
fig = go.Figure()
fig.add_trace(go.Scatterpolar(
    r=e_data, theta=angles_e, name="E-Plane", 
    line=dict(color='#58a6ff', width=2),
    fill='toself', fillcolor='rgba(88, 166, 255, 0.1)'
))
fig.add_trace(go.Scatterpolar(
    r=h_data, theta=angles_e, name="H-Plane", 
    line=dict(color='#ff7b72', width=2),
    fill='toself', fillcolor='rgba(255, 123, 114, 0.1)'
))

fig.update_layout(
    title=dict(text="Real-time HFSS Radiation Pattern", font=dict(color='#c9d1d9')),
    height=500,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#8b949e', family='Inter, sans-serif'),
    polar=dict(
        radialaxis=dict(visible=True, showline=False, gridcolor='#30363d', tickfont=dict(color='#8b949e')),
        angularaxis=dict(gridcolor='#30363d', tickfont=dict(color='#c9d1d9')),
        bgcolor='#0d1117'
    ),
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

time.sleep(2)
st.rerun()