import http.server
import json
import math
import os
import random
import sys
import threading
import time
from datetime import datetime

try:
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    HAS_ML = True
except ImportError:
    HAS_ML = False

HTTP_PORT = 8000
state_lock = threading.Lock()

state = {
    "esp32_connected": False,
    "node_a_online": True,
    "node_b_online": True,
    "rssi_a": -70.0,
    "rssi_b": -75.0,
    "uav_angle_deg": 45,
    "gain_a_db": 0.0,
    "gain_b_db": 0.0,
    "active_link": "NODE_A",
    "ai_confidence": 0.85,
    "decision_mode": "SPATIAL_ML",
    "uptime_s": 0,
    "antenna_data_loaded": False,
    "polar_data": {"angles": [], "e_plane": [], "h_plane": []},
    "logs": []
}

ml_model = None
antenna_lookup = {}

def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] [{level}] {msg}"
    print(entry)
    with state_lock:
        state["logs"].append(entry)
        if len(state["logs"]) > 50:
            state["logs"].pop(0)

def load_antenna_csv():
    global antenna_lookup
    csv_file = "NetOne_farfield_base.csv"
    if not os.path.exists(csv_file):
        log(f"Antenna CSV '{csv_file}' not found. Using default omni gain.", "WARN")
        return

    log(f"Loading HFSS far-field dataset from {csv_file}...", "INFO")
    try:
        import pandas as pd
        df = pd.read_csv(csv_file)
        
        # Populate angle-to-gain lookup table
        for _, row in df.iterrows():
            p, t, g = int(row['Phi[deg]']), int(row['Theta[deg]']), float(row['GainTotal'])
            g_db = 10 * math.log10(g + 1e-10)
            antenna_lookup[(p, t)] = g_db

        # Extract E-Plane (Phi=0) and H-Plane (Phi=90) slice for UI visualization
        phi_0 = df[df['Phi[deg]'] == 0].sort_values('Theta[deg]')
        phi_90 = df[df['Phi[deg]'] == 90].sort_values('Theta[deg]')
        
        angles = phi_0['Theta[deg]'].tolist()
        e_plane = [round(10 * math.log10(g + 1e-10), 2) for g in phi_0['GainTotal']]
        h_plane = [round(10 * math.log10(g + 1e-10), 2) for g in phi_90['GainTotal']]

        with state_lock:
            state["polar_data"] = {"angles": angles, "e_plane": e_plane, "h_plane": h_plane}
            state["antenna_data_loaded"] = True
        log("HFSS Antenna data parsed successfully.", "INFO")
    except Exception as e:
        log(f"Failed to parse antenna CSV: {e}", "ERROR")

def get_gain(phi, theta):
    p = max(0, min(360, int(phi)))
    t = max(0, min(180, int(theta)))
    return antenna_lookup.get((p, t), -10.0)

def train_spatial_model():
    global ml_model
    if not HAS_ML:
        return
    log("Training Spatial-Aware Random Forest Engine...", "INFO")
    X, y = [], []
    for _ in range(1000):
        hour = random.randint(0, 23)
        rssi_a = random.uniform(-95, -50)
        rssi_b = random.uniform(-95, -50)
        gain_a = random.uniform(-25, 0)
        gain_b = random.uniform(-25, 0)
        
        # Total link budget score considering antenna pattern
        score_a = (rssi_a + gain_a + 120) * (1.2 if 8 <= hour <= 17 else 0.8)
        score_b = (rssi_b + gain_b + 120)
        target = 0 if score_a >= score_b else 1
        
        X.append([hour, rssi_a, rssi_b, gain_a, gain_b])
        y.append(target)
    
    clf = RandomForestRegressor(n_estimators=25, random_state=42)
    clf.fit(X, y)
    ml_model = clf
    log("Spatial ML Model fit complete.", "INFO")

def ai_engine():
    while True:
        time.sleep(1.0)
        with state_lock:
            rssi_a = state["rssi_a"]
            rssi_b = state["rssi_b"]
            angle = state["uav_angle_deg"]
            hour = datetime.now().hour
            
            # Obtain gain relative to current UAV angular displacement
            gain_a = get_gain(0, angle)
            gain_b = get_gain(90, angle)
            
            state["gain_a_db"] = round(gain_a, 2)
            state["gain_b_db"] = round(gain_b, 2)

            if ml_model and HAS_ML:
                pred = ml_model.predict([[hour, rssi_a, rssi_b, gain_a, gain_b]])[0]
                confidence = abs(pred - 0.5) * 2.0
                chosen = "NODE_A" if pred < 0.5 else "NODE_B"
                mode = "SPATIAL_ML"
            else:
                chosen = "NODE_A" if (rssi_a + gain_a) >= (rssi_b + gain_b) else "NODE_B"
                confidence = 0.90
                mode = "DIRECT_GAINED"

            state["active_link"] = chosen
            state["ai_confidence"] = round(confidence, 2)
            state["decision_mode"] = mode

def simulate_telemetry():
    while True:
        time.sleep(2.0)
        with state_lock:
            if not state["esp32_connected"]:
                hour = datetime.now().hour
                base_a = -65.0 if 8 <= hour <= 17 else -85.0
                base_b = -85.0 if 8 <= hour <= 17 else -65.0
                
                state["rssi_a"] = round(base_a + random.uniform(-5, 5), 1)
                state["rssi_b"] = round(base_b + random.uniform(-5, 5), 1)
                state["uav_angle_deg"] = (state["uav_angle_deg"] + 5) % 180
                state["uptime_s"] += 2

class DIAOHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with state_lock:
                payload = json.dumps(state)
            self.wfile.write(payload.encode("utf-8"))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            html = """<!DOCTYPE html>
<html>
<head>
    <title>DIAO AI Telemetry & Antenna Dashboard</title>
    <style>
        body { font-family: monospace; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .card { background: #1e293b; border: 1px solid #334155; padding: 15px; border-radius: 8px; }
        .val { font-size: 1.6em; font-weight: bold; color: #38bdf8; margin-top: 5px; }
        .main-container { display: grid; grid-template-columns: 1fr 320px; gap: 20px; }
        #logs { background: #020617; border: 1px solid #1e293b; padding: 10px; height: 260px; overflow-y: auto; border-radius: 6px; font-size: 0.85em; }
        canvas { background: #1e293b; border: 1px solid #334155; border-radius: 8px; width: 100%; height: 300px; }
    </style>
</head>
<body>
    <h2>DIAO Spatial AI Bridge & Antenna Telemetry</h2>
    <div class="grid">
        <div class="card"><div>ACTIVE LINK</div><div id="active" class="val">--</div></div>
        <div class="card"><div>NODE A RSSI / GAIN</div><div id="node_a_val" class="val">--</div></div>
        <div class="card"><div>NODE B RSSI / GAIN</div><div id="node_b_val" class="val">--</div></div>
        <div class="card"><div>DECISION ENGINE</div><div id="engine" class="val">--</div></div>
    </div>
    
    <div class="main-container">
        <div class="card">
            <div>SYSTEM LOGS</div>
            <div id="logs"></div>
        </div>
        <div>
            <canvas id="polarCanvas" width="300" height="300"></canvas>
        </div>
    </div>

    <script>
        let polarDataLoaded = false;
        
        function drawPolar(angles, ePlane, currentAngle) {
            const canvas = document.getElementById('polarCanvas');
            const ctx = canvas.getContext('2d');
            const w = canvas.width;
            const h = canvas.height;
            const cx = w / 2;
            const cy = h / 2;
            const radius = 110;

            ctx.clearRect(0, 0, w, h);

            // Draw polar grid circles
            ctx.strokeStyle = '#334155';
            ctx.lineWidth = 1;
            [0.3, 0.6, 1.0].forEach(r => {
                ctx.beginPath();
                ctx.arc(cx, cy, radius * r, 0, 2 * Math.PI);
                ctx.stroke();
            });

            // Draw axis lines
            ctx.beginPath();
            ctx.moveTo(cx - radius, cy); ctx.lineTo(cx + radius, cy);
            ctx.moveTo(cx, cy - radius); ctx.lineTo(cx, cy + radius);
            ctx.stroke();

            if (!angles || angles.length === 0) return;

            // Plot E-Plane Pattern
            ctx.beginPath();
            ctx.strokeStyle = '#38bdf8';
            ctx.lineWidth = 2;
            
            for (let i = 0; i < angles.length; i++) {
                const rad = (angles[i] - 90) * (Math.PI / 180);
                const normVal = Math.max(0, (ePlane[i] + 30) / 30);
                const r = radius * normVal;
                const x = cx + r * Math.cos(rad);
                const y = cy + r * Math.sin(rad);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Plot current UAV Position Vector
            const uavRad = (currentAngle - 90) * (Math.PI / 180);
            ctx.beginPath();
            ctx.strokeStyle = '#fbbf24';
            ctx.lineWidth = 2;
            ctx.moveTo(cx, cy);
            ctx.lineTo(cx + radius * Math.cos(uavRad), cy + radius * Math.sin(uavRad));
            ctx.stroke();
            
            ctx.fillStyle = '#f8fafc';
            ctx.font = '12px monospace';
            ctx.fillText('Live Radiation Pattern', 10, 20);
        }

        async function update() {
            try {
                const res = await fetch('/api/data');
                const data = await res.json();
                document.getElementById('active').innerText = data.active_link;
                document.getElementById('node_a_val').innerText = data.rssi_a + ' dBm (' + data.gain_a_db + ' dBi)';
                document.getElementById('node_b_val').innerText = data.rssi_b + ' dBm (' + data.gain_b_db + ' dBi)';
                document.getElementById('engine').innerText = data.decision_mode + ' (' + Math.round(data.ai_confidence * 100) + '%)';
                
                const logBox = document.getElementById('logs');
                logBox.innerHTML = data.logs.join('<br>');
                logBox.scrollTop = logBox.scrollHeight;

                if (data.antenna_data_loaded) {
                    drawPolar(data.polar_data.angles, data.polar_data.e_plane, data.uav_angle_deg);
                }
            } catch(e) {}
        }
        setInterval(update, 1000);
        update();
    </script>
</body>
</html>"""
            self.wfile.write(html.encode("utf-8"))

    def log_message(self, format, *args):
        return

if __name__ == "__main__":
    print("==================================================")
    print(" DIAO SPATIAL AI BRIDGE STARTING...")
    print("==================================================")
    load_antenna_csv()
    train_spatial_model()
    
    threading.Thread(target=ai_engine, daemon=True).start()
    threading.Thread(target=simulate_telemetry, daemon=True).start()
    
    server = http.server.HTTPServer(('0.0.0.0', HTTP_PORT), DIAOHandler)
    print(f"[OK] Dashboard active at http://localhost:{HTTP_PORT}")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped.")
        sys.exit(0)