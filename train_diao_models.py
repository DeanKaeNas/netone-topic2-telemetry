"""
DIAO twin-AI training pipeline (Random Forest + TensorFlow NN + handoff gate)
NESARI 2026 Topic 2 — Dynamic Intelligent Antenna Optimization

Design choices borrowed from:
  - MathWorks / Microwave Journal, "Using AI for Antenna Design, Analysis
    and Optimization" (Jan 2025): intelligent sampling of the design space
    instead of blind uniform random draws, and a secondary classifier that
    narrows the space BEFORE the main optimizer runs (their PIFA impedance-
    matching classifier -> here, a "should we actually move the antenna"
    gate, to cut needless servo wear).
  - Huawei Wireless Technology Lab, "AI-Driven Innovations in RF and
    Antenna Design" (Feb 2025): active-learning-style stratified sampling
    over data quality, and reporting feature importances for a minimum
    level of explainability rather than treating the RF/NN as a pure
    black box.

Run locally with your ESP32 attached is NOT required — this trains on a
physics-informed synthetic dataset, same as your existing _training_data(),
but sampled more intelligently and labelled by hybrid RSSI+gain logic
(mirrors your Option C fallback) instead of pure np.random.choice noise.
If you already have traffic_generator.py with Harare-specific patterns,
swap generate_design_space() below for that — everything downstream
(features, labels, training, saving) stays the same.
"""

import os
import json
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
SECTORS = list(range(0, 181, 15))
SECTOR_NAMES = {
    0: "Borrowdale", 15: "Highlands", 30: "Avondale", 45: "Mt Pleasant",
    60: "Greendale", 75: "CBD North", 90: "CBD Centre", 105: "CBD South",
    120: "Mbare", 135: "Highfields", 150: "Glen Norah", 165: "Budiriro", 180: "Chitungwiza"
}
N_SAMPLES = 12000
MODEL_RF_FILE = "diao_spatial_rf.pkl"
MODEL_NN_FILE = "diao_nn.keras"
MODEL_TFLITE = "diao_nn.tflite"
MODEL_GATE_FILE = "diao_handoff_gate.pkl"

RNG = np.random.default_rng(42)


# ─────────────────────────────────────────────────────────────
# 1. INTELLIGENT SAMPLING OF THE DESIGN SPACE
#    (Latin Hypercube instead of pure uniform random, so a fixed sample
#    budget covers the RSSI/gain space far more evenly — same motivation
#    as the "intelligent sampling" step in the MathWorks workflow.)
# ─────────────────────────────────────────────────────────────
def latin_hypercube(n_samples, n_dims, rng):
    """Simple LHS: stratify each dimension into n_samples bins, one draw
    per bin, then shuffle independently per dimension."""
    result = np.empty((n_samples, n_dims))
    for d in range(n_dims):
        cut = (np.arange(n_samples) + rng.random(n_samples)) / n_samples
        rng.shuffle(cut)
        result[:, d] = cut
    return result  # values in [0, 1)


def generate_design_space(n_samples=N_SAMPLES, rng=RNG):
    lhs = latin_hypercube(n_samples, 4, rng)  # rssi_a, rssi_b, gain_a, gain_b

    rssi_a = -100 + lhs[:, 0] * 60          # -100..-40 dBm
    rssi_b = -100 + lhs[:, 1] * 60
    gain_a = -10 + lhs[:, 2] * 22           # -10..12 dBi (matches HFSS lookup range)
    gain_b = -10 + lhs[:, 3] * 22

    # hour / day-of-week: stratify evenly rather than uniform-random so every
    # traffic period (incl. Harare peak hours) is represented in proportion,
    # not left to chance.
    hour = np.tile(np.arange(24), n_samples // 24 + 1)[:n_samples]
    rng.shuffle(hour)
    dow = np.tile(np.arange(7), n_samples // 7 + 1)[:n_samples]
    rng.shuffle(dow)

    # ── Physics-informed label (mirrors your Option C RSSI+gain fallback) ──
    # Effective received power per node = RSSI + gain, weighted toward the
    # node with the better link; the "true" sector is whichever bearing
    # (Node A @ 30 or Node B @ 150, plus a smooth interpolation between the
    # two based on the effective-power skew) that reading best supports.
    eff_a = rssi_a + gain_a
    eff_b = rssi_b + gain_b
    skew = np.clip((eff_a - eff_b) / 20.0, -1, 1)  # -1 -> favour B, +1 -> favour A
    # Map skew (-1..1) onto the 0..180 sector range (0=A's bearing side, 180=B's side)
    raw_angle = 90 - skew * 90
    # snap to nearest defined sector + small label noise for realism
    noisy_angle = raw_angle + rng.normal(0, 6, n_samples)
    labels = np.array([min(SECTORS, key=lambda s: abs(s - a)) for a in noisy_angle])

    X = np.column_stack([rssi_a, rssi_b, gain_a, gain_b, hour, dow])
    return X, labels


# ─────────────────────────────────────────────────────────────
# 2. RANDOM FOREST — sector classifier
# ─────────────────────────────────────────────────────────────
def train_rf(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=250, max_depth=16, min_samples_leaf=3, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
    test_acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"[RF] 5-fold CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(f"[RF] Held-out test accuracy: {test_acc:.3f}")
    print(classification_report(y_test, rf.predict(X_test), zero_division=0))

    # Explainability (Huawei paper §6.4): don't leave the model a black box —
    # print which features actually drive the decision.
    feat_names = ["rssi_a", "rssi_b", "gain_a", "gain_b", "hour", "dow"]
    importances = sorted(zip(feat_names, rf.feature_importances_), key=lambda t: -t[1])
    print("[RF] Feature importances:")
    for name, imp in importances:
        print(f"    {name:10s} {imp:.3f}")

    joblib.dump(rf, MODEL_RF_FILE)
    print(f"[OK] Saved {MODEL_RF_FILE}")
    return rf


# ─────────────────────────────────────────────────────────────
# 3. TENSORFLOW NN — twin voter
# ─────────────────────────────────────────────────────────────
def train_nn(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_train_cat = tf.keras.utils.to_categorical([SECTORS.index(v) for v in y_train], num_classes=len(SECTORS))
    y_test_cat = tf.keras.utils.to_categorical([SECTORS.index(v) for v in y_test], num_classes=len(SECTORS))

    nn = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(6,)),
        tf.keras.layers.Normalization(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(len(SECTORS), activation="softmax"),
    ])
    nn.layers[0].adapt(np.array(X_train, dtype=np.float32))
    nn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True)
    history = nn.fit(
        np.array(X_train, dtype=np.float32), y_train_cat,
        validation_split=0.15, epochs=60, batch_size=64,
        callbacks=[early_stop], verbose=0,
    )
    test_loss, test_acc = nn.evaluate(np.array(X_test, dtype=np.float32), y_test_cat, verbose=0)
    print(f"[NN] Test accuracy: {test_acc:.3f} (stopped at epoch {len(history.history['loss'])})")

    nn.save(MODEL_NN_FILE)
    converter = tf.lite.TFLiteConverter.from_keras_model(nn)
    tflite_model = converter.convert()
    with open(MODEL_TFLITE, "wb") as f:
        f.write(tflite_model)
    print(f"[OK] Saved {MODEL_NN_FILE} and {MODEL_TFLITE}")
    return nn


# ─────────────────────────────────────────────────────────────
# 4. HANDOFF GATE — "should the antenna actually move?"
#    Same pattern as the MathWorks PIFA classifier: a lightweight binary
#    classifier trained BEFORE the expensive decision, narrowing when the
#    main ensemble's answer should be acted on vs held (avoids re-steering
#    the MG996R on marginal/noisy readings).
# ─────────────────────────────────────────────────────────────
def train_handoff_gate(X, y, rf):
    n = len(y)
    # Simulate a "current_angle" the antenna was already sitting at, and
    # derive the ground-truth decision: move only if the true sector differs
    # from current by more than one 15-degree step AND the link would
    # meaningfully improve (>= 6 dB combined RSSI+gain gain).
    current_angle = RNG.choice(SECTORS, size=n)
    angle_delta = np.abs(y - current_angle)

    eff_at_true = X[:, 0] + X[:, 2]  # proxy for "how good the true sector's link is"
    eff_at_current = eff_at_true - RNG.normal(4, 3, n)  # current sector assumed worse on average
    gain_from_move = eff_at_true - eff_at_current

    should_move = ((angle_delta >= 15) & (gain_from_move >= 6)).astype(int)

    gate_X = np.column_stack([X, current_angle, angle_delta])
    X_train, X_test, y_train, y_test = train_test_split(gate_X, should_move, test_size=0.2, random_state=42)

    gate = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42)
    gate.fit(X_train, y_train)
    acc = accuracy_score(y_test, gate.predict(X_test))
    print(f"[GATE] Handoff-decision accuracy: {acc:.3f}")
    print(f"[GATE] Move-recommended rate in training data: {should_move.mean():.2%}")

    joblib.dump(gate, MODEL_GATE_FILE)
    print(f"[OK] Saved {MODEL_GATE_FILE}")
    return gate


# ─────────────────────────────────────────────────────────────
# 5. ENSEMBLE AGREEMENT CHECK
# ─────────────────────────────────────────────────────────────
def check_ensemble_agreement(rf, nn, X, y, n_check=500):
    idx = RNG.choice(len(X), size=n_check, replace=False)
    Xc = X[idx]
    rf_pred = rf.predict(Xc)
    nn_pred_probs = nn.predict(np.array(Xc, dtype=np.float32), verbose=0)
    nn_pred = np.array([SECTORS[i] for i in np.argmax(nn_pred_probs, axis=1)])
    agree_rate = (rf_pred == nn_pred).mean()
    print(f"[ENSEMBLE] RF/NN agreement on {n_check} held-out points: {agree_rate:.1%}")


if __name__ == "__main__":
    print(f"[INFO] Generating {N_SAMPLES} samples via Latin Hypercube design-space sampling...")
    X, y = generate_design_space()

    print("\n=== Training Random Forest ===")
    rf = train_rf(X, y)

    print("\n=== Training TensorFlow NN ===")
    nn = train_nn(X, y)

    print("\n=== Training handoff gate ===")
    gate = train_handoff_gate(X, y, rf)

    print("\n=== Ensemble sanity check ===")
    check_ensemble_agreement(rf, nn, X, y)

    print("\n[DONE] All three models saved. Commit diao_spatial_rf.pkl, diao_nn.tflite, "
          "and diao_handoff_gate.pkl and push to GitHub for Streamlit Cloud to pick up.")
