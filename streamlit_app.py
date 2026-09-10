import json
import time
import random
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import paho.mqtt.client as mqtt
import tflite_runtime.interpreter as tflite
import plotly.graph_objects as go

st.set_page_config(page_title="DIAO — NetOne Intelligence", layout="wide")

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
        st.session_state.telemetry = json.loads(msg.payload)
    except Exception:
        pass

if "telemetry" not in st.session_state:
    st.session_state.telemetry = None
    st.session_state.sim_rssi_a = -72.0
    st.session_state.sim_rssi_b = -85.0
    try:
        mqtt_client = mqtt.Client(client_id="diao_streamlit_subscriber")
        mqtt_client.on_message = on_mqtt_message
        mqtt_client.connect(BROKER, 1883, 60)
        mqtt_client.subscribe(TOPIC)
        mqtt_client.loop_start()
    except Exception:
        pass

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

current_hour = datetime.now().hour
current_dow = datetime.now().weekday()
gain_a = get_antenna_gain(0, 60) + random.uniform(-1, 1)
gain_b = get_antenna_gain(90, 60) + random.uniform(-1, 1)

# Ensemble Model Evaluation
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

# UI Display
st.title("DIAO — Dynamic Intelligent Antenna Optimization")
status_label = "[LIVE (MQTT)]" if is_live else "[SIMULATION - Bridge Offline]"
st.caption(f"NetOne Band 3 · 1800 MHz · Status: {status_label}")

if is_live and telemetry_data:
    esp_status = "Connected" if telemetry_data.get("esp32_connected") else "Simulation Mode"
    st.caption(f"ESP32: {esp_status} | Decision Reason: {telemetry_data.get('decision_reason', 'N/A')}")

col1, col2, col3, col4, col5, col6 = st.columns(6)
nearest_sector = min(SECTORS, key=lambda s: abs(s - current_angle))
col1.metric("Current Angle", f"{current_angle}°", SECTOR_NAMES.get(nearest_sector, "—"))
col2.metric("AI Prediction", f"{ai_angle}°", f"{ensemble_confidence:.0%} confidence")
col3.metric("Node A RSSI", f"{rssi_a:.0f} dBm")
col4.metric("Node B RSSI", f"{rssi_b:.0f} dBm")
col5.metric("RF Confidence", f"{rf_confidence:.0%}")
col6.metric("NN Confidence", f"{nn_confidence:.0%}")

fig = go.Figure()
fig.add_trace(go.Scatterpolar(r=e_data, theta=angles_e, name="E-Plane"))
fig.add_trace(go.Scatterpolar(r=h_data, theta=angles_e, name="H-Plane"))
fig.update_layout(title="HFSS Radiation Pattern (E/H Plane)", height=420, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

time.sleep(2)
st.rerun()