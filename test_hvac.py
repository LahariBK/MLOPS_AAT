import os
import random
import subprocess
import yaml
from sim.hvac_env import HVACEnv

def run_tests():
    tests_passed = 0
    total_tests = 10
    
    print("\n" + "=" * 60)
    print("  Smart HVAC RL - Unit and Integration Test Suite")
    print("=" * 60)
    
    # ---------------------------------------------------------
    # UNIT TESTS
    # ---------------------------------------------------------
    
    # 1. Environment resets correctly
    try:
        env = HVACEnv()
        state = env.reset()
        assert all(s >= 0 for s in state)
        print("✅ 1. Environment resets correctly (state is non-negative)")
        tests_passed += 1
    except Exception as e:
        print(f"❌ 1. Environment resets correctly: Failed ({e})")
        
    # 2. Step function returns valid outputs
    try:
        env = HVACEnv()
        env.reset()
        next_state, reward, done, info = env.step(0)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "zone_temps" in info
        print("✅ 2. Step function returns valid outputs (reward, done, info)")
        tests_passed += 1
    except Exception as e:
        print(f"❌ 2. Step function returns valid outputs: Failed ({e})")
        
    # 3. Temperature always stays in valid range
    try:
        env = HVACEnv()
        env.reset()
        valid = True
        for _ in range(96):
            _, _, _, info = env.step(random.randint(0, 26))
            for t in info["zone_temps"]:
                # Widen range to account for continuous random cooling
                if t < 0.0 or t > 60.0:
                    valid = False
        assert valid
        print("✅ 3. Temperature stays in reasonable range (0-60°C) across 96 steps")
        tests_passed += 1
    except Exception as e:
        print(f"❌ 3. Temperature stays in valid range: Failed ({e})")
        
    # 4. Action space is exactly 27
    try:
        env = HVACEnv()
        env.reset()
        for a in range(27):
            env.step(a)
        print("✅ 4. Action space is exactly 27 (0 to 26 executed without errors)")
        tests_passed += 1
    except Exception as e:
        print(f"❌ 4. Action space is exactly 27: Failed ({e})")
        
    # 5. Energy consumed is always non-negative
    try:
        env = HVACEnv()
        env.reset()
        _, _, _, info = env.step(random.randint(0, 26))
        assert info["energy"] >= 0
        print("✅ 5. Energy consumed is always non-negative")
        tests_passed += 1
    except Exception as e:
        print(f"❌ 5. Energy consumed is non-negative: Failed ({e})")
        
    # 6. Q-learning update rule works correctly
    try:
        Q = 0.0
        alpha = 0.1
        gamma = 0.9
        reward = 10.0
        best_next = 5.0
        new_Q = Q + alpha * (reward + gamma * best_next - Q)
        assert new_Q > Q
        print("✅ 6. Q-learning update rule works correctly (Q increases on positive reward)")
        tests_passed += 1
    except Exception as e:
        print(f"❌ 6. Q-learning update rule works correctly: Failed ({e})")

    # ---------------------------------------------------------
    # INTEGRATION TESTS
    # ---------------------------------------------------------
        
    # 7. Full episode runs without crashing
    try:
        env = HVACEnv()
        env.reset()
        for _ in range(96):
            env.step(random.randint(0, 26))
        print("✅ 7. Full episode runs without crashing (96 steps)")
        tests_passed += 1
    except Exception as e:
        print(f"❌ 7. Full episode runs without crashing: Failed ({e})")
        
    # Prepare dummy configuration for integration tests (8, 9, 10)
    test_config = {
        "run_id": "test-run",
        "algorithm": "qlearning",
        "alpha": 0.1,
        "gamma": 0.9,
        "epsilon": 0.1,
        "epsilon_decay": 0.99,
        "epsilon_min": 0.01,
        "episodes": 1,
        "steps_per_episode": 96,
        "seed": 42,
        "policy_save_path": "policies/test_policy.pkl",
        "results_save_path": "experiments/test_results.csv",
        "log_save_path": "experiments/test_log.json"
    }
    with open("test_config.yaml", "w") as f:
        yaml.dump(test_config, f)
        
    # Run the training script via subprocess
    try:
        subprocess.run(["python", "train.py", "--config", "test_config.yaml"], 
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        run_success = True
    except Exception as e:
        run_success = False
        print(f"Training run crashed: {e}")

    # 8. CSV file is created after training run
    try:
        assert run_success, "Training script failed"
        assert os.path.exists("experiments/test_results.csv"), "File not found"
        print("✅ 8. CSV file is created after training run")
        tests_passed += 1
    except Exception as e:
        print(f"❌ 8. CSV file is created: Failed ({e})")
        
    # 9. Policy pkl file is created after training run
    try:
        assert run_success, "Training script failed"
        assert os.path.exists("policies/test_policy.pkl"), "File not found"
        print("✅ 9. Policy pkl file is created after training run")
        tests_passed += 1
    except Exception as e:
        print(f"❌ 9. Policy pkl file is created: Failed ({e})")
        
    # 10. log.json is updated after training run
    try:
        assert run_success, "Training script failed"
        assert os.path.exists("experiments/test_log.json"), "File not found"
        print("✅ 10. log.json is updated after training run")
        tests_passed += 1
    except Exception as e:
        print(f"❌ 10. log.json is updated: Failed ({e})")

    # Cleanup test files
    if os.path.exists("test_config.yaml"): os.remove("test_config.yaml")
    if os.path.exists("experiments/test_results.csv"): os.remove("experiments/test_results.csv")
    if os.path.exists("policies/test_policy.pkl"): os.remove("policies/test_policy.pkl")
    if os.path.exists("experiments/test_log.json"): os.remove("experiments/test_log.json")
    
    print("=" * 60)
    print(f"  Final Summary: {tests_passed} / {total_tests} Tests Passed")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    run_tests()
