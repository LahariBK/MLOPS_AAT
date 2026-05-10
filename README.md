# Smart HVAC Reinforcement Learning + MLOps

## Problem Statement
Building climate control represents a major source of global energy consumption. Traditional HVAC systems rely on fixed timers or simple thermostatic rules, leading to significant energy waste during unoccupied periods and suboptimal comfort. This project implements a Reinforcement Learning (RL) agent that intelligently controls a multi-zone HVAC system. By learning the thermal dynamics and occupancy schedules of different zones, the agent optimizes for human comfort while drastically reducing energy usage. 

## Sustainable Development Goals (SDGs)
This project directly contributes to:
- **SDG 7 (Affordable and Clean Energy):** By dynamically turning down HVAC power when zones are unoccupied and intelligently pre-cooling, the system minimizes energy waste, promoting efficiency.
- **SDG 11 (Sustainable Cities and Communities):** Implementing smart, efficient building management at scale builds resilient, sustainable urban infrastructure while improving the quality of life for inhabitants.

## Folder Structure
```
Smart_HVAC_RL/
├── sim/
│   ├── __init__.py
│   └── hvac_env.py        # HVACEnv class simulator
├── configs/
│   ├── qlearning_v1.yaml  # Config for run 1
│   └── qlearning_v2.yaml  # Config for run 2 (exploration focus)
├── experiments/           # Auto-created tracking logs and CSVs
├── policies/              # Auto-created serialized Q-tables
├── train.py               # Reproducible training entry point
├── evaluate.py            # Baseline vs RL evaluation script
├── plot_results.py        # Visualization script
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

## Algorithm Choice & Definitions
**Algorithm:** Tabular Q-learning
**Justification:** Tabular Q-learning was chosen because the problem can be discretized into a relatively small, finite state space. It guarantees convergence to the optimal policy, avoids the instability and computational overhead of Deep RL, and allows for extremely fast training and inference, which is ideal for this AAT evaluation.

**State Space:** 
A tuple: `(zone_0_temp_bin, zone_1_temp_bin, zone_2_temp_bin, hour_of_day)`. Temperatures are binned into 1°C increments between 20°C and 30°C. Total states: `11 * 11 * 11 * 24 = 31,944`.

**Action Space:** 
Joint per-zone control with 3 discrete settings: `(OFF, LOW, HIGH)^3 = 27` possible actions.
- OFF: 0 energy, 0 cooling.
- LOW: 1 unit energy, moderate cooling.
- HIGH: 2.5 units energy, strong cooling.

**Reward Function:**
- **Comfort:** +10 per zone in the comfort band (21–24°C) when occupied.
- **Energy Penalty:** Proportional to the energy consumed by the action.
- **Deviation Penalty:** If occupied, large penalty for deviation from 21-24°C. If unoccupied, smaller penalty for deviation from a wider 18-28°C band.

## Exact Reproduce Commands
To completely reproduce the results, run the following commands sequentially:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the agent using config v1 (this auto-creates tags and policies)
python train.py --config configs/qlearning_v1.yaml

# 3. Train the agent using config v2
python train.py --config configs/qlearning_v2.yaml

# 4. Evaluate the trained agent against the baseline
python evaluate.py

# 5. Generate plots for the first run
python plot_results.py --results experiments/results_1.csv
```

## Experiment Tracking Explanation
The project uses custom, lightweight MLOps tracking:
1. **Configuration:** YAML files define all hyperparameters, ensuring full reproducibility.
2. **Metrics:** `train.py` tracks rewards, temperatures, and energy per episode, exporting them to `experiments/results_X.csv`.
3. **Registry:** A central `experiments/log.json` maintains a registry of all runs, their configurations, and their final performance summaries.
4. **Versioning:** The code leverages `git` to automatically commit and tag the repository with the `run_id` (e.g., `exp-qlearning-1`) upon completion of a training run.

### Git Tag Table
| Tag | Run ID | Purpose |
|-----|--------|---------|
| `exp-qlearning-1` | `exp-qlearning-1` | Standard Q-learning training run (epsilon=0.3) |
| `exp-qlearning-2` | `exp-qlearning-2` | High exploration run (epsilon=0.5) |

### Policy Versions Table
| Policy File | Config Source | Characteristics |
|-------------|---------------|-----------------|
| `policy_v1.pkl` | `configs/qlearning_v1.yaml` | Standard learned policy. |
| `policy_v2_explored.pkl` | `configs/qlearning_v2.yaml` | Explored policy, potentially better handling of edge states. |

## Monitoring Plan
To deploy this system in a real-world production environment, the following 6 metrics must be monitored continuously:
1. **Avg Zone Temp:** Track average temperatures during occupied hours to ensure they stay within the 21-24°C comfort band.
2. **Energy / Hour:** Monitor the energy consumption rate to calculate cost savings and detect potential hardware failures (e.g., AC stuck on HIGH).
3. **Occupancy Accuracy:** Compare the scheduled occupancy against actual PIR/motion sensor data. Discrepancies mean the schedule needs updating.
4. **ε-decay Health:** During online learning or retraining, ensure the exploration rate (epsilon) decays according to the mathematical schedule and does not get stuck.
5. **Safety Overrides:** Track how often human occupants manually override the thermostat. Frequent overrides indicate the policy is failing to meet comfort needs.
6. **Policy Drift Detection:** Monitor the distribution of states encountered. If seasonal weather shifts significantly, the state distribution will drift, triggering a retraining pipeline.
