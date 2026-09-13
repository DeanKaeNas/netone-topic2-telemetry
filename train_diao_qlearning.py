"""
DIAO Q-learning layer
NESARI 2026 Topic 2 — Dynamic Intelligent Antenna Optimization (DIAO)

Purpose
-------
train_diao_models.py trains RF + NN to IMITATE a hand-written label
(the RSSI+gain skew formula). This module instead learns by INTERACTION:
the agent picks a sector, observes a reward describing how good that
choice was, and updates its value estimates from that feedback — the
same supervised/unsupervised distinction the AI-Driven Innovations in
RF and Antenna Design paper draws, and what your own proposal's
Objective 2 / Phase 3 commits to (reinforcement learning, not just
classification).

Design choices, and why
------------------------
- TABULAR Q-learning, not Deep Q-Networks: converges in minutes on a
  laptop with no GPU, has an inspectable Q-table you can print and
  defend in a viva, and carries none of the "undertrained deep policy
  moves the live servo somewhere wrong in front of judges" risk that a
  DQN/PPO agent would this close to a deadline.
- State = (current sector, RSSI bucket, time-of-day bucket) — coarse on
  purpose. Fine-grained continuous state is what the NN is already for;
  the Q-learning layer's job is to learn WHEN a move is worth it, not
  to re-derive fine antenna physics.
- Reward is shaped from the same RSSI+gain skew formula used to label
  the supervised data (generate_design_space in train_diao_models.py),
  but the agent never sees that "true" target directly — it only sees
  the reward number after acting, same as it would from a real RSRP
  measurement post-move on hardware. This is the qualitative
  difference that makes this actually RL rather than another
  classifier trained on the same labels.

Integration
------------
This module is ADDITIVE. It does not replace the RF/NN ensemble that
already runs safely in streamlit_app.py — it runs alongside it as a
second, independently-learned opinion, shown to the viewer but not
(yet) given control authority. See the dashboard's new "Learned policy"
line. Promote it to authoritative once you're confident in its
behaviour on real telemetry, by swapping the read-only display for an
actual vote in the ensemble decision block.
"""

import os
import json
import numpy as np
import joblib

SECTORS = list(range(0, 181, 15))
N_SECTORS = len(SECTORS)

# Bucketed on the EFFECTIVE POWER DIFFERENCE between the two nodes
# (rssi+gain for A minus rssi+gain for B) — this is what actually
# determines the target angle in the shaped reward below, so it is
# the signal the agent needs in its state. Bucketing on raw RSSI alone
# (an earlier version of this script did that) discards exactly the
# information the reward depends on, making the environment look
# unlearnable — a real bug, not just noise, caught by the sanity check
# in evaluate() below.
DIFF_EDGES = [-1000, -20, -12, -4, 4, 12, 20, 1000]  # 7 buckets
N_DIFF_BUCKETS = len(DIFF_EDGES) - 1
N_HOUR_BUCKETS = 6                         # 4-hour bins; not yet used by the reward,
                                            # kept as a hook for future traffic-pattern shaping

MOVE_PENALTY = 0.04       # discourages needless servo wear, mirrors the handoff gate's intent
ALPHA = 0.15              # learning rate
GAMMA = 0.90              # discount factor
EPISODES = 20000
EPS_START, EPS_END = 1.0, 0.05

Q_TABLE_FILE = "diao_qtable.pkl"
RNG = np.random.default_rng(7)


# ─────────────────────────────────────────────────────────────
# Discretization
# ─────────────────────────────────────────────────────────────
def sector_idx(angle):
    return min(range(N_SECTORS), key=lambda i: abs(SECTORS[i] - angle))

def diff_bucket(eff_diff):
    idx = np.digitize([eff_diff], DIFF_EDGES)[0] - 1
    return int(np.clip(idx, 0, N_DIFF_BUCKETS - 1))

def hour_bucket(hour):
    return int(hour // (24 // N_HOUR_BUCKETS)) % N_HOUR_BUCKETS

def state_index(sec_i, diff_i, hour_i):
    return (sec_i * N_DIFF_BUCKETS + diff_i) * N_HOUR_BUCKETS + hour_i

N_STATES = N_SECTORS * N_DIFF_BUCKETS * N_HOUR_BUCKETS


# ─────────────────────────────────────────────────────────────
# Simulated environment
# One step = one telemetry reading + one antenna-steering decision.
# The "true" target angle (skew formula) is used only to SHAPE the
# reward — the agent is never told it directly, only the reward number.
# ─────────────────────────────────────────────────────────────
class DIAOEnv:
    def __init__(self, rng):
        self.rng = rng
        self.reset()

    def reset(self):
        self.current_angle = int(self.rng.choice(SECTORS))
        self._resample_conditions()
        return self._state()

    def _resample_conditions(self):
        self.rssi_a = self.rng.uniform(-100, -40)
        self.rssi_b = self.rng.uniform(-100, -40)
        self.gain_a = self.rng.uniform(-10, 12)
        self.gain_b = self.rng.uniform(-10, 12)
        self.hour = int(self.rng.integers(0, 24))

    def _target_angle(self):
        eff_a = self.rssi_a + self.gain_a
        eff_b = self.rssi_b + self.gain_b
        skew = np.clip((eff_a - eff_b) / 20.0, -1, 1)
        return 90 - skew * 90

    def _state(self):
        eff_diff = (self.rssi_a + self.gain_a) - (self.rssi_b + self.gain_b)
        return state_index(
            sector_idx(self.current_angle),
            diff_bucket(eff_diff),
            hour_bucket(self.hour),
        )

    def step(self, action_angle):
        target = self._target_angle()
        angular_error = abs(action_angle - target) / 180.0
        moved = action_angle != self.current_angle
        reward = -angular_error - (MOVE_PENALTY if moved else 0.0)

        self.current_angle = action_angle
        self._resample_conditions()  # conditions drift for the next reading
        return self._state(), reward


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────
def train():
    env = DIAOEnv(RNG)
    Q = np.zeros((N_STATES, N_SECTORS))
    reward_history = []

    for ep in range(EPISODES):
        eps = EPS_START + (EPS_END - EPS_START) * (ep / EPISODES)
        s = env._state()

        if RNG.random() < eps:
            a_idx = RNG.integers(0, N_SECTORS)
        else:
            a_idx = int(np.argmax(Q[s]))
        action_angle = SECTORS[a_idx]

        s_next, reward = env.step(action_angle)
        Q[s, a_idx] += ALPHA * (reward + GAMMA * np.max(Q[s_next]) - Q[s, a_idx])
        reward_history.append(reward)

        if (ep + 1) % 4000 == 0:
            avg_r = np.mean(reward_history[-4000:])
            print(f"[Q-LEARN] episode {ep+1}/{EPISODES}  avg reward (last 4000): {avg_r:.4f}  eps={eps:.2f}")

    return Q


def evaluate(Q, n_checks=2000):
    """Greedy-policy sanity check against the same shaped target used in training."""
    env = DIAOEnv(np.random.default_rng(99))
    errors = []
    for _ in range(n_checks):
        s = env._state()
        a_idx = int(np.argmax(Q[s]))
        action_angle = SECTORS[a_idx]
        target = env._target_angle()
        errors.append(abs(action_angle - target))
        env.step(action_angle)
    mae = np.mean(errors)
    print(f"[Q-LEARN] Greedy-policy mean angular error vs shaped target: {mae:.1f} degrees")
    return mae


def save(Q):
    payload = {
        "q_table": Q,
        "sectors": SECTORS,
        "diff_edges": DIFF_EDGES,
        "n_hour_buckets": N_HOUR_BUCKETS,
    }
    joblib.dump(payload, Q_TABLE_FILE)
    print(f"[OK] Saved {Q_TABLE_FILE}  (shape={Q.shape}, {Q.nbytes/1024:.0f} KB)")


# ─────────────────────────────────────────────────────────────
# Inference helper — import this from the dashboard
# ─────────────────────────────────────────────────────────────
def q_recommend(current_angle, rssi_a, gain_a, rssi_b, gain_b, hour, q_payload=None):
    """Returns (recommended_angle, confidence 0-1) from a loaded Q-table payload."""
    if q_payload is None:
        q_payload = joblib.load(Q_TABLE_FILE)
    Q = q_payload["q_table"]
    eff_diff = (rssi_a + gain_a) - (rssi_b + gain_b)
    s = state_index(
        sector_idx(current_angle),
        diff_bucket(eff_diff),
        hour_bucket(hour),
    )
    q_row = Q[s]
    a_idx = int(np.argmax(q_row))
    # softmax over the row as a rough confidence score
    exp_q = np.exp(q_row - np.max(q_row))
    probs = exp_q / exp_q.sum()
    return SECTORS[a_idx], float(probs[a_idx])


if __name__ == "__main__":
    print(f"[INFO] Training tabular Q-learning agent — {N_STATES} states x {N_SECTORS} actions, {EPISODES} episodes")
    Q = train()
    evaluate(Q)
    save(Q)
    print("[DONE] Commit diao_qtable.pkl and push — the dashboard will pick it up automatically "
          "(and skip it gracefully if absent).")
