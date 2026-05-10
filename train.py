import argparse
import yaml
import json
import csv
import os
import pickle
import numpy as np
import subprocess
from sim.hvac_env import HVACEnv

def set_seed(seed):
    np.random.seed(seed)

def run_git_command(command):
    try:
        subprocess.run(command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e.stderr.decode('utf-8')}")

def main():
    parser = argparse.ArgumentParser(description="Train HVAC RL Agent")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    
    # Create directories if they don't exist
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("policies", exist_ok=True)

    env = HVACEnv()

    # Algorithm: Q-learning (tabular)
    # Justification: Tabular Q-learning is chosen because the state space is discretized and 
    # relatively small (11 temperature bins per zone * 3 zones * 24 hours = 31,944 states).
    # The action space is also small (3 actions per zone ^ 3 zones = 27 actions). 
    # This guarantees convergence to the optimal policy without the instability or computational 
    # overhead of function approximation (e.g., Deep RL), making it highly suitable for 
    # reliable evaluation and training within resource constraints.
    
    action_space_size = 27
    q_table = {}

    def get_q(state, action):
        if state not in q_table:
            q_table[state] = np.zeros(action_space_size)
        return q_table[state][action]

    def get_max_q(state):
        if state not in q_table:
            q_table[state] = np.zeros(action_space_size)
        return np.max(q_table[state])

    def get_best_action(state):
        if state not in q_table:
            q_table[state] = np.zeros(action_space_size)
        # Handle ties randomly instead of always choosing index 0
        q_values = q_table[state]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return np.random.choice(best_actions)

    alpha = config["alpha"]
    gamma = config["gamma"]
    epsilon = config["epsilon"]
    epsilon_decay = config["epsilon_decay"]
    epsilon_min = config["epsilon_min"]

    results = []

    for episode in range(config["episodes"]):
        state = env.reset()
        total_reward = 0
        episode_energy = 0
        zone_temps_history = []
        
        for step in range(config["steps_per_episode"]):
            if np.random.rand() < epsilon:
                action = np.random.randint(action_space_size)
            else:
                action = get_best_action(state)

            next_state, reward, done, info = env.step(action)
            
            # Update Q-table
            current_q = get_q(state, action)
            max_next_q = get_max_q(next_state)
            new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
            q_table[state][action] = new_q
            
            state = next_state
            total_reward += reward
            episode_energy += info["energy"]
            zone_temps_history.append(info["zone_temps"])

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        avg_temps = np.mean(zone_temps_history, axis=0)
        avg_energy = episode_energy / config["steps_per_episode"]
        
        results.append({
            "run_id": config["run_id"],
            "episode": episode + 1,
            "total_reward": total_reward,
            "avg_temp_zone0": avg_temps[0],
            "avg_temp_zone1": avg_temps[1],
            "avg_temp_zone2": avg_temps[2],
            "avg_energy": avg_energy,
            "epsilon": epsilon
        })
        
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}/{config['episodes']} - Reward: {total_reward:.2f} - Epsilon: {epsilon:.3f}")

    # Save policy
    with open(config["output_policy"], "wb") as f:
        pickle.dump(q_table, f)

    # Write results CSV
    csv_file = config["output_results"]
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Append to log.json
    log_file = config["log_file"]
    log_data = []
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            try:
                log_data = json.load(f)
            except json.JSONDecodeError:
                pass
                
    summary = {
        "run_id": config["run_id"],
        "episodes": config["episodes"],
        "final_epsilon": epsilon,
        "final_avg_reward": np.mean([r["total_reward"] for r in results[-10:]]),
        "policy_path": config["output_policy"],
        "results_path": config["output_results"]
    }
    log_data.append(summary)
    
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=4)
        
    print(f"Run {config['run_id']} completed. Results saved.")

    # MLOps: Auto-commit and git tag
    print("Performing auto-commit and tagging...")
    run_git_command("git add .")
    run_git_command(f'git commit -m "Auto-commit for run {config["run_id"]}"')
    # If tag exists, delete it first or handle it. Let's force update the tag.
    run_git_command(f'git tag -f {config["run_id"]}')
    print(f"Tagged repository with {config['run_id']}")

if __name__ == "__main__":
    main()
