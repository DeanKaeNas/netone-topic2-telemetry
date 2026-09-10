import os

# Suppress TensorFlow verbose logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import json
import threading
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# Configuration & File Paths
MODEL_RF_FILE = "diao_spatial_rf.pkl"
MODEL_NN_FILE = "diao_nn.keras"
MODEL_TFLITE  = "diao_nn.tflite"
MQTT_BROKER   = "broker.hivemq.com"
MQTT_TOPIC    = "diao/netone/telemetry"

SECTORS = ["Sector_A", "Sector_B", "Sector_C", "Sector_D"]

# Imports with fallback checks
try:
    import paho.mqtt.client as mqtt
    HAS_MQTT = True
except ImportError:
    HAS_MQTT = False
    print("[WARN] paho-mqtt not installed")

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("[WARN] tensorflow not installed")

rf_model = None
nn_model = None

def _training_data(n_samples=10000):
    X = np.random.uniform(low=-100.0, high=0.0, size=(n_samples, 6))
    y = np.random.choice(SECTORS, size=n_samples)
    return X, y

def load_or_train():
    global rf_model, nn_model
    
    # 1. Random Forest Model
    if os.path.exists(MODEL_RF_FILE):
        try:
            rf_model = joblib.load(MODEL_RF_FILE)
            print(f"[OK] Loaded RF Model: {MODEL_RF_FILE}")
        except Exception as e:
            print(f"[ERROR] Loading RF model failed: {e}")

    if rf_model is None:
        print("[INFO] Training Random Forest model...")
        X, y = _training_data(10000)
        rf_model = RandomForestClassifier(n_estimators=200, max_depth=14, random_state=42, n_jobs=-1)
        rf_model.fit(X, y)
        joblib.dump(rf_model, MODEL_RF_FILE)
        print(f"[OK] Saved RF Model: {MODEL_RF_FILE}")

    # 2. TensorFlow Neural Network
    if os.path.exists(MODEL_NN_FILE):
        try:
            nn_model = tf.keras.models.load_model(MODEL_NN_FILE)
            print(f"[OK] Loaded NN Model: {MODEL_NN_FILE}")
            return
        except Exception as e:
            print(f"[ERROR] Loading NN model failed: {e}")

    if not HAS_TF:
        print("[WARN] TensorFlow not available. Skipping NN training.")
        return

    print("[INFO] Training TensorFlow Sector Classifier...")
    X, y = _training_data(10000)
    Xa = np.array(X, dtype=np.float32)
    ya = tf.keras.utils.to_categorical([SECTORS.index(v) for v in y], num_classes=len(SECTORS))

    nn_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(6,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(len(SECTORS), activation="softmax")
    ])
    nn_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    nn_model.fit(Xa, ya, epochs=30, batch_size=64, verbose=0)

    # Save Keras format (Local) and TFLite format (Streamlit Cloud)
    nn_model.save(MODEL_NN_FILE)
    converter = tf.lite.TFLiteConverter.from_keras_model(nn_model)
    tflite_model = converter.convert()
    with open(MODEL_TFLITE, "wb") as f:
        f.write(tflite_model)
    print(f"[OK] Saved NN Models: {MODEL_NN_FILE} and {MODEL_TFLITE}")

def mqtt_publisher():
    if not HAS_MQTT:
        return
    try:
        client = mqtt.Client(client_id="diao_main_publisher")
        client.connect(MQTT_BROKER, 1883, 60)
        client.loop_start()
        print(f"[OK] MQTT Publisher connected to {MQTT_BROKER}")
    except Exception as e:
        print(f"[WARN] MQTT connection failed: {e}")
        return

    keys_to_send = [
        "current_angle", "ai_angle", "rssi_a", "rssi_b", "confidence",
        "coverage_gain", "node_a_online", "node_b_online", "packets",
        "decision_reason", "esp32_connected"
    ]

    while True:
        time.sleep(2)
        try:
            with lock:
                payload = {k: state[k] for k in keys_to_send if k in state}
            payload["ts"] = int(time.time())
            client.publish(MQTT_TOPIC, json.dumps(payload), qos=0)
        except Exception:
            pass

if __name__ == "__main__":
    load_or_train()
    threading.Thread(target=mqtt_publisher, daemon=True).start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Application terminated.")