import numpy as np
from datetime import datetime, timedelta
import os
import pickle


# Utility function to handle datetime strings consistently
def parse_datetime(dt_str):
    if dt_str is None:
        return None
    # Check if the string is offset-aware (has a Z or timezone)
    if isinstance(dt_str, str):
        if "Z" in dt_str or "+" in dt_str:
            # Make offset-aware datetime
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        else:
            # Make naive datetime
            return datetime.fromisoformat(dt_str)
    return dt_str  # Already a datetime object


class ParkingEnvironment:
    def __init__(self, num_bays=4, slots_per_bay=10):
        self.num_bays = num_bays
        self.slots_per_bay = slots_per_bay
        self.state_size = self.num_bays * self.slots_per_bay
        self.action_size = self.state_size
        self.q_table = np.zeros(
            (4, 24, self.action_size)
        )  # 4 priority levels, 24 hours
        self.parking_lot = np.zeros(
            (num_bays, slots_per_bay)
        )  # 0 = empty, 1 = occupied
        self.current_hour = 0  # Default hour

    def reset(self, occupancy_rate=0.5, hour=None):
        # Initialize with some random occupancy
        self.parking_lot = np.random.choice(
            [0, 1],
            size=(self.num_bays, self.slots_per_bay),
            p=[1 - occupancy_rate, occupancy_rate],
        )
        self.state = self.parking_lot.flatten()
        # Set current hour for time-dependent decisions
        self.current_hour = hour if hour is not None else np.random.randint(0, 24)
        return self.state

    def get_available_spaces(self):
        # Return indices of available spaces
        return np.where(self.state == 0)[0]

    def step(self, action, priority):
        reward = -100  # Default negative reward for invalid actions
        done = False

        # Check if the action (parking space) is available
        if 0 <= action < self.state_size and self.state[action] == 0:
            bay = action // self.slots_per_bay
            slot = action % self.slots_per_bay

            # Reward structure with time factor:
            # - Higher priority vehicles get better rewards for closer spots
            # - Lower slots (closer to entrance) are more valuable
            # - Different bays may have different base values
            # - Time-of-day affects reward (e.g., premium spots more valuable during peak hours)
            bay_value = 10 - bay  # Bay 0 is most valuable
            slot_value = self.slots_per_bay - slot  # Slot 0 is most valuable
            priority_factor = priority + 1

            # Time factor - peak hours (8-10am, 5-7pm) get higher rewards for good spots
            peak_hours = [8, 9, 10, 17, 18, 19]
            time_factor = 1.5 if self.current_hour in peak_hours else 1.0

            # Calculate reward
            reward = priority_factor * (bay_value + slot_value) * time_factor

            # Mark the spot as occupied
            self.state[action] = 1
            self.parking_lot[bay, slot] = 1

        # Check if the parking lot is full
        if np.all(self.state == 1):
            done = True

        return self.state, reward, done

    def get_bay_slot(self, action):
        bay = action // self.slots_per_bay
        slot = action % self.slots_per_bay
        return bay + 1, slot + 1  # Adding 1 to match 1-based indexing


class SmartParkingSystem:
    def __init__(self, score_model, bay_model, slot_model, env):
        self.score_model = score_model
        self.bay_model = bay_model
        self.slot_model = slot_model
        self.env = env

    def predict_allocation_score(self, vehicle_data):
        # Extract features including temporal ones
        if isinstance(vehicle_data.get("arrival_time"), str):
            arrival_time = parse_datetime(
                vehicle_data.get("arrival_time", "2023-05-01 08:00:00")
            )
            departure_time = parse_datetime(
                vehicle_data.get("departure_time", "2023-05-01 12:00:00")
            )
        else:
            arrival_time = vehicle_data.get("arrival_time", datetime.now())
            departure_time = vehicle_data.get(
                "departure_time", arrival_time + timedelta(hours=2)
            )

        # Calculate time features
        arrival_hour = arrival_time.hour
        arrival_minute = arrival_time.minute
        arrival_day_of_week = arrival_time.weekday()
        stay_duration_minutes = (departure_time - arrival_time).total_seconds() / 60

        features = np.array(
            [
                [
                    vehicle_data.get("vehicle_plate_type", 0),
                    vehicle_data.get("vehicle_type", 0),
                    vehicle_data.get("day_of_week", arrival_day_of_week),
                    vehicle_data.get("hour_of_day", arrival_hour),
                    vehicle_data.get("duration_hours", stay_duration_minutes / 60),
                    vehicle_data.get("priority_level", 0),
                    arrival_hour,
                    arrival_minute,
                    arrival_day_of_week,
                    stay_duration_minutes,
                ]
            ]
        )

        # Predict allocation score
        predicted_score = self.score_model.predict(features)[0]
        return float(predicted_score)  # Convert numpy.float64 to Python float

    def allocate_parking(self, vehicle_data):
        # First, predict the allocation score
        predicted_score = self.predict_allocation_score(vehicle_data)

        # Get priority level and time
        priority = vehicle_data.get("priority_level", 0)

        # Get current hour from the vehicle data or use a default
        if isinstance(vehicle_data.get("arrival_time"), str):
            arrival_time = parse_datetime(
                vehicle_data.get("arrival_time", "2023-05-01 08:00:00")
            )
            current_hour = arrival_time.hour
        else:
            current_hour = vehicle_data.get(
                "hour_of_day", vehicle_data.get("arrival_time", datetime.now()).hour
            )

        # Reset environment with current hour and realistic occupancy
        # Higher occupancy during peak hours
        if current_hour in [8, 9, 17, 18]:  # Peak hours
            occupancy_rate = 0.7
        elif current_hour in [7, 10, 16, 19]:  # Near peak
            occupancy_rate = 0.5
        else:
            occupancy_rate = 0.3

        self.env.reset(occupancy_rate=occupancy_rate, hour=current_hour)

        # Use Q-learning for decision
        available_spaces = self.env.get_available_spaces()

        if len(available_spaces) == 0:
            return {"status": "Parking full", "allocation_score": predicted_score}

        # Use Q-table to select best action for this priority level and hour
        priority_hour_q_values = self.env.q_table[priority, current_hour]

        # Filter Q-values to only consider available spaces
        available_q_values = [
            (action, priority_hour_q_values[action]) for action in available_spaces
        ]

        # Sort by Q-value (highest first)
        sorted_actions = sorted(available_q_values, key=lambda x: x[1], reverse=True)
        best_action = sorted_actions[0][0]

        # Take the action
        _, reward, _ = self.env.step(best_action, priority)

        # Get bay and slot numbers
        bay, slot = self.env.get_bay_slot(best_action)

        # Format arrival_time for output
        if isinstance(vehicle_data.get("arrival_time"), str):
            allocation_time = vehicle_data.get("arrival_time")
        else:
            allocation_time = vehicle_data.get("arrival_time", datetime.now()).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        return {
            "status": "Allocated",
            "bay_assigned": int(bay),  # Convert numpy types to Python native types
            "slot_assigned": int(slot),
            "priority_level": (
                int(priority) if isinstance(priority, np.integer) else priority
            ),
            "allocation_score": float(predicted_score),
            "q_value": float(sorted_actions[0][1]),
            "reward": float(reward),
            "allocation_time": allocation_time,
            "parking_map": self.visualize_parking_map(),
        }

    def visualize_parking_map(self):
        """Generate a text representation of the current parking lot state"""
        result = "Parking Lot Status:\n"
        result += "Legend: [Empty: □, Occupied: ■]\n\n"

        for bay in range(self.env.num_bays):
            result += f"Bay {bay+1}: "
            for slot in range(self.env.slots_per_bay):
                if self.env.parking_lot[bay, slot] == 0:
                    result += "□ "  # Empty
                else:
                    result += "■ "  # Occupied
            result += "\n"

        return result
