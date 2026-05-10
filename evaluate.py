import argparse
import pickle
import numpy as np
from sim.hvac_env import HVACEnv

def get_best_action(q_table, state):
    action_space_size = 27
    if state not in q_table:
        return 0 # Default to OFF if unseen state
    q_values = q_table[state]
    max_q = np.max(q_values)
    best_actions = np.where(q_values == max_q)[0]
    return np.random.choice(best_actions)

def run_episode(env, policy_func, steps=24):
    state = env.reset()
    total_reward = 0
    total_energy = 0
    total_discomfort = 0
    
    for _ in range(steps):
        action = policy_func(state, env.hour)
        next_state, reward, _, info = env.step(action)
        
        total_reward += reward
        total_energy += info["energy"]
        total_discomfort += info["discomfort"]
        state = next_state
        
    return total_reward, total_energy, total_discomfort

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL policy vs Baseline")
    parser.add_argument("--policy", type=str, default="policies/policy_v1.pkl", help="Path to policy pkl file")
    args = parser.parse_args()

    # Load trained policy
    policy_path = args.policy
    try:
        with open(policy_path, "rb") as f:
            q_table = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {policy_path} not found. Please run train.py first.")
        return

    env = HVACEnv()
    
    # Define RL policy wrapper
    def rl_policy(state, hour):
        return get_best_action(q_table, state)
        
    # Define Baseline policy (Fixed-timer)
    # Turn AC on LOW (action 1 for each zone) when occupied, else OFF
    def baseline_policy(state, hour):
        acts = [0, 0, 0]
        # Same occupancy rules as env
        if 18 <= hour <= 23: acts[0] = 1
        if hour >= 22 or hour <= 8: acts[1] = 1
        if 9 <= hour <= 17: acts[2] = 1
        
        # Encode to action_idx
        action_idx = acts[0] + acts[1]*3 + acts[2]*9
        return action_idx

    # Evaluate over multiple episodes (days) to get reliable averages
    num_eval_episodes = 30
    np.random.seed(42)
    
    rl_rewards, rl_energies, rl_discomforts = [], [], []
    base_rewards, base_energies, base_discomforts = [], [], []
    
    for _ in range(num_eval_episodes):
        r, e, d = run_episode(env, rl_policy)
        rl_rewards.append(r)
        rl_energies.append(e)
        rl_discomforts.append(d)
        
        r, e, d = run_episode(env, baseline_policy)
        base_rewards.append(r)
        base_energies.append(e)
        base_discomforts.append(d)
        
    avg_rl_reward = np.mean(rl_rewards)
    avg_rl_energy = np.mean(rl_energies)
    avg_rl_discomfort = np.mean(rl_discomforts)
    
    avg_base_reward = np.mean(base_rewards)
    avg_base_energy = np.mean(base_energies)
    avg_base_discomfort = np.mean(base_discomforts)
    
    # Calculate % improvements
    # Higher reward is better
    imp_reward = ((avg_rl_reward - avg_base_reward) / max(0.1, abs(avg_base_reward))) * 100
    # Lower energy is better
    imp_energy = ((avg_base_energy - avg_rl_energy) / max(0.1, avg_base_energy)) * 100
    # Lower discomfort is better
    imp_discomfort = ((avg_base_discomfort - avg_rl_discomfort) / max(0.1, avg_base_discomfort)) * 100

    print("\n====================================================")
    print("  METRIC                 FIXED-TIMER    RL-POLICY")
    print("----------------------------------------------------")
    print(f"  Avg episode reward         {avg_base_reward:<14.2f}{avg_rl_reward:<14.2f}")
    print(f"  Total energy/episode       {avg_base_energy:<14.2f}{avg_rl_energy:<14.2f}")
    print(f"  Discomfort steps/ep        {avg_base_discomfort:<14.2f}{avg_rl_discomfort:<14.2f}")
    print("====================================================")
    print(f"  Energy saving      : {imp_energy:+.2f}%")
    print(f"  Comfort improvement: {imp_discomfort:+.2f}%")
    print("\n  SDG 7 Impact: {:.2f}% energy reduction supports".format(imp_energy))
    print("  affordable, clean energy use in buildings.")
    print("\n  SDG 11 Impact: {:.2f}% fewer discomfort steps".format(imp_discomfort))
    print("  supports sustainable, liveable urban environments.")

if __name__ == "__main__":
    main()
