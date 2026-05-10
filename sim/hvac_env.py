import numpy as np

class HVACEnv:
    def __init__(self):
        # 3 Zones: 0: Living Room, 1: Bedroom, 2: Office
        self.num_zones = 3
        
        # Thermal resistance (how slowly they match outdoor temp)
        # Lower means it matches outdoor temp faster
        self.thermal_resistance = [0.1, 0.08, 0.12]
        
        # Cooling power of actions (0: OFF, 1: LOW, 2: HIGH)
        self.action_effects = [0.0, -1.5, -3.0]
        self.energy_cost = [0.0, 1.0, 2.5] # Cost per step for OFF/LOW/HIGH
        
        # Time variables
        self.hour = 0
        self.day = 0
        
        # State variables (Current temperatures)
        self.zone_temps = np.array([25.0, 25.0, 25.0])
        
    def reset(self):
        self.hour = 0
        self.day = 0
        # Start at a warm morning temperature
        self.zone_temps = np.array([25.0, 25.0, 25.0])
        return self._get_state()

    def _get_outdoor_temp(self, hour):
        # Min at 4 AM (25C), Max at 4 PM (35C)
        return 30.0 - 5.0 * np.cos(2 * np.pi * (hour - 4) / 24.0)

    def _is_occupied(self, zone, hour):
        if zone == 0:   # Living Room: 18:00 - 23:00
            return 18 <= hour <= 23
        elif zone == 1: # Bedroom: 22:00 - 08:00
            return hour >= 22 or hour <= 8
        elif zone == 2: # Office: 09:00 - 17:00
            return 9 <= hour <= 17
        return False

    def _get_state(self):
        # State: (zone_0_bin, zone_1_bin, zone_2_bin, hour_of_day)
        state = []
        for temp in self.zone_temps:
            # Bin temperatures from 20 to 30 into integers
            temp_bin = int(np.clip(np.round(temp), 20, 30))
            state.append(temp_bin)
        state.append(self.hour)
        return tuple(state)

    def step(self, action_idx):
        """
        action_idx is an integer from 0 to 26
        """
        # Decode joint action
        z0_act = action_idx % 3
        z1_act = (action_idx // 3) % 3
        z2_act = (action_idx // 9) % 3
        acts = [z0_act, z1_act, z2_act]
        
        outdoor_temp = self._get_outdoor_temp(self.hour)
        
        reward = 0
        energy_used = 0
        discomfort_steps = 0
        
        for i in range(self.num_zones):
            # Thermal dynamics
            heat_gain = self.thermal_resistance[i] * (outdoor_temp - self.zone_temps[i])
            cooling = self.action_effects[acts[i]]
            self.zone_temps[i] += heat_gain + cooling
            
            # Energy penalty
            energy = self.energy_cost[acts[i]]
            energy_used += energy
            reward -= energy * 2.0  # Energy penalty multiplier
            
            # Comfort constraints and rewards
            occupied = self._is_occupied(i, self.hour)
            temp = self.zone_temps[i]
            
            if occupied:
                if 21.0 <= temp <= 24.0:
                    reward += 10.0 # Reward for comfort when occupied
                else:
                    # Deviation penalty
                    deviation = min(abs(temp - 21.0), abs(temp - 24.0)) if temp < 21.0 or temp > 24.0 else 0
                    reward -= deviation * 3.0
                    discomfort_steps += 1
            else:
                # Unoccupied: Wider comfort band (e.g., 18 to 28)
                if temp < 18.0 or temp > 28.0:
                    deviation = min(abs(temp - 18.0), abs(temp - 28.0))
                    reward -= deviation * 1.0

        # Advance time
        self.hour += 1
        if self.hour >= 24:
            self.hour = 0
            self.day += 1
            
        next_state = self._get_state()
        done = False # Continuous episodic learning handled externally
        
        info = {
            'outdoor_temp': outdoor_temp,
            'zone_temps': self.zone_temps.copy(),
            'energy': energy_used,
            'discomfort': discomfort_steps,
            'actions': acts
        }
        
        return next_state, reward, done, info
