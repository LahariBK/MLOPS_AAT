import argparse
import csv
import json
import os
import pickle
import random
import subprocess
import time
from pathlib import Path

import mlflow
import numpy as np
import yaml

from sim.hvac_env import HVACEnv


# ──────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train HVAC RL agent")
parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

print(f"\n{'='*55}")
print(f"  Smart HVAC RL Trainer")
print(f"  Run ID : {cfg['run_id']}")
print(f"  Config : {args.config}")
print(f"{'='*55}\n")

# ──────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────
SEED = cfg["seed"]
random.seed(SEED)
np.random.seed(SEED)

# ──────────────────────────────────────────────────────────────
# ENVIRONMENT + Q-TABLE
# ──────────────────────────────────────────────────────────────
env = HVACEnv()
n_actions = 27

def encode_state(state):
    t0, t1, t2, hour = state
    t0 = int(np.clip(t0 - 15, 0, 10))
    t1 = int(np.clip(t1 - 15, 0, 10))
    t2 = int(np.clip(t2 - 15, 0, 10))
    return t0 * 11 * 11 * 24 + t1 * 11 * 24 + t2 * 24 + int(hour)

n_states = 11 * 11 * 11 * 24
Q = np.zeros((n_states, n_actions))

alpha     = cfg["alpha"]
gamma     = cfg["gamma"]
epsilon   = cfg["epsilon"]
eps_decay = cfg["epsilon_decay"]
eps_min   = cfg["epsilon_min"]

episodes     = cfg["episodes"]
steps_per_ep = cfg["steps_per_episode"]

# ──────────────────────────────────────────────────────────────
# OUTPUT PATHS
# ──────────────────────────────────────────────────────────────
Path("policies").mkdir(exist_ok=True)
Path("experiments").mkdir(exist_ok=True)

policy_path  = cfg["policy_save_path"]
results_path = cfg["results_save_path"]
log_path     = cfg["log_save_path"]

# ──────────────────────────────────────────────────────────────
# MLOPS: MLflow Setup
# ──────────────────────────────────────────────────────────────
mlflow.set_experiment("Smart_HVAC_RL")
mlflow.start_run(run_name=cfg["run_id"])
mlflow.log_params({
    "algorithm":     cfg["algorithm"],
    "alpha":         alpha,
    "gamma":         gamma,
    "epsilon":       epsilon,
    "epsilon_decay": eps_decay,
    "epsilon_min":   eps_min,
    "episodes":      episodes,
    "seed":          SEED,
})

# ──────────────────────────────────────────────────────────────
# MLOPS: CSV Log Setup
# ──────────────────────────────────────────────────────────────
csv_file = open(results_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "run_id", "episode", "total_reward",
    "avg_temp_zone0", "avg_temp_zone1", "avg_temp_zone2",
    "avg_energy", "epsilon"
])

start_time = time.time()
rewards_per_episode = []
energy_per_episode  = []

# ──────────────────────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────────────────────
print(f"Training for {episodes} episodes...\n")

for ep in range(episodes):
    state        = env.reset()
    total_reward = 0.0
    total_energy = 0.0
    temp_log     = [[], [], []]

    for _ in range(steps_per_ep):
        s_idx = encode_state(state)

        # epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, n_actions - 1)
        else:
            action = int(np.argmax(Q[s_idx]))

        next_state, reward, done, info = env.step(action)
        ns_idx = encode_state(next_state)

        # Q-learning update
        best_next = np.max(Q[ns_idx])
        Q[s_idx, action] += alpha * (reward + gamma * best_next - Q[s_idx, action])

        state         = next_state
        total_reward += reward
        total_energy += info["energy"]
        for z in range(3):
            temp_log[z].append(info["zone_temps"][z])

        if done:
            break

    # Decay epsilon
    epsilon = max(eps_min, epsilon * eps_decay)

    rewards_per_episode.append(total_reward)
    energy_per_episode.append(total_energy)

    avg_temps  = [np.mean(temp_log[z]) for z in range(3)]
    avg_energy = total_energy / steps_per_ep

    # MLflow metric logging per episode
    mlflow.log_metric("total_reward",   total_reward,  step=ep)
    mlflow.log_metric("epsilon",        epsilon,       step=ep)
    mlflow.log_metric("avg_energy",     avg_energy,    step=ep)
    mlflow.log_metric("avg_temp_zone0", avg_temps[0],  step=ep)
    mlflow.log_metric("avg_temp_zone1", avg_temps[1],  step=ep)
    mlflow.log_metric("avg_temp_zone2", avg_temps[2],  step=ep)

    # CSV logging per episode
    csv_writer.writerow([
        cfg["run_id"], ep + 1, round(total_reward, 2),
        round(avg_temps[0], 2), round(avg_temps[1], 2), round(avg_temps[2], 2),
        round(avg_energy, 3), round(epsilon, 4)
    ])

    if (ep + 1) % 10 == 0:
        print(f"  Ep {ep+1:>4}/{episodes}  |  "
              f"Reward: {total_reward:>8.1f}  |  "
              f"e: {epsilon:.3f}  |  "
              f"Avg Temps: {avg_temps[0]:.1f}  {avg_temps[1]:.1f}  {avg_temps[2]:.1f}")

csv_file.close()
elapsed = time.time() - start_time

# ──────────────────────────────────────────────────────────────
# MLOPS: Save Policy
# ──────────────────────────────────────────────────────────────
with open(policy_path, "wb") as f:
    pickle.dump({"Q": Q, "config": cfg}, f)
mlflow.log_artifact(policy_path)
print(f"\n  Policy saved -> {policy_path}")

# ──────────────────────────────────────────────────────────────
# MLOPS: Save JSON Log
# ──────────────────────────────────────────────────────────────
log_entry = {
    "run_id":      cfg["run_id"],
    "config_file": args.config,
    "algorithm":   cfg["algorithm"],
    "parameters": {
        "alpha":         alpha,
        "gamma":         gamma,
        "epsilon_start": cfg["epsilon"],
        "epsilon_decay": eps_decay,
        "epsilon_min":   eps_min,
    },
    "training": {
        "episodes":          episodes,
        "steps_per_episode": steps_per_ep,
        "seed":              SEED,
    },
    "results": {
        "avg_reward_last_100":  round(float(np.mean(rewards_per_episode[-100:])), 2),
        "avg_reward_first_100": round(float(np.mean(rewards_per_episode[:100])), 2),
        "best_episode_reward":  round(float(max(rewards_per_episode)), 2),
        "avg_energy_last_100":  round(float(np.mean(energy_per_episode[-100:])), 3),
    },
    "elapsed_seconds": round(elapsed, 1),
    "policy_path":     policy_path,
    "results_csv":     results_path,
}

existing = []
if os.path.exists(log_path):
    with open(log_path) as f:
        existing = json.load(f)
existing.append(log_entry)
with open(log_path, "w") as f:
    json.dump(existing, f, indent=2)

mlflow.log_artifact(log_path)
mlflow.log_artifact(results_path)

# Final summary metrics
mlflow.log_metric("avg_reward_last_100",  log_entry["results"]["avg_reward_last_100"])
mlflow.log_metric("avg_reward_first_100", log_entry["results"]["avg_reward_first_100"])
mlflow.log_metric("best_episode_reward",  log_entry["results"]["best_episode_reward"])
mlflow.log_metric("avg_energy_last_100",  log_entry["results"]["avg_energy_last_100"])

mlflow.end_run()

print(f"  Log saved    -> {log_path}")
print(f"  Results CSV  -> {results_path}")
print(f"  MLflow run   -> experiment: Smart_HVAC_RL | run: {cfg['run_id']}")

# ──────────────────────────────────────────────────────────────
# MLOPS: Git tag this experiment
# ──────────────────────────────────────────────────────────────
try:
    tag = cfg["run_id"]
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-m", f"MLOps: {tag} - training complete"], check=True)
    subprocess.run(["git", "tag", "-f", tag], check=True)
    print(f"  Git tag      -> {tag}")
except Exception as e:
    print(f"  Git tagging skipped: {e}")

print(f"\n{'='*55}")
print(f"  Training complete in {elapsed:.1f}s")
print(f"  Avg reward (last 100 eps): {log_entry['results']['avg_reward_last_100']}")
print(f"{'='*55}\n")