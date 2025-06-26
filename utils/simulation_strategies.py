import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from utils.parking_system import ParkingEnvironment, SmartParkingSystem


class AllocationStrategy:
    """Base class for parking allocation strategies"""

    def __init__(self, num_bays: int = 4, slots_per_bay: int = 10):
        self.num_bays = num_bays
        self.slots_per_bay = slots_per_bay
        self.total_slots = num_bays * slots_per_bay
        self.reset_environment()

    def reset_environment(self):
        """Reset the parking environment to empty state"""
        self.parking_lot = np.zeros((self.num_bays, self.slots_per_bay))
        self.occupancy_map = np.zeros(self.total_slots)

    def get_available_slots(self) -> List[int]:
        """Get list of available slot indices"""
        return [i for i in range(self.total_slots) if self.occupancy_map[i] == 0]

    def get_bay_slot_from_index(self, slot_index: int) -> Tuple[int, int]:
        """Convert slot index to bay and slot numbers (1-based)"""
        bay = (slot_index // self.slots_per_bay) + 1
        slot = (slot_index % self.slots_per_bay) + 1
        return bay, slot

    def mark_slot_occupied(self, slot_index: int):
        """Mark a slot as occupied"""
        self.occupancy_map[slot_index] = 1
        bay_idx = slot_index // self.slots_per_bay
        slot_idx = slot_index % self.slots_per_bay
        self.parking_lot[bay_idx, slot_idx] = 1

    def allocate_slot(self, vehicle_data: Dict) -> Dict:
        """Base allocation method to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement allocate_slot method")


class SequentialAllocationStrategy(AllocationStrategy):
    """Sequential allocation: assigns to the first available slot"""

    def allocate_slot(self, vehicle_data: Dict) -> Dict:
        available_slots = self.get_available_slots()

        if not available_slots:
            return {
                "status": "Parking full",
                "allocation_score": 0.0,
                "allocation_time": datetime.now().isoformat(),
            }

        # Get the first available slot (sequential)
        selected_slot = available_slots[0]
        bay, slot = self.get_bay_slot_from_index(selected_slot)

        # Mark slot as occupied
        self.mark_slot_occupied(selected_slot)

        # Calculate a simple score based on slot position
        allocation_score = float(100 - selected_slot)  # Higher score for earlier slots

        return {
            "status": "Allocated",
            "bay_assigned": bay,
            "slot_assigned": slot,
            "allocation_score": allocation_score,
            "allocation_time": datetime.now().isoformat(),
        }


class RandomAllocationStrategy(AllocationStrategy):
    """Random allocation: assigns to a random available slot"""

    def __init__(self, num_bays: int = 4, slots_per_bay: int = 10, seed: int = None):
        super().__init__(num_bays, slots_per_bay)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def allocate_slot(self, vehicle_data: Dict) -> Dict:
        available_slots = self.get_available_slots()

        if not available_slots:
            return {
                "status": "Parking full",
                "allocation_score": 0.0,
                "allocation_time": datetime.now().isoformat(),
            }

        # Select a random available slot
        selected_slot = random.choice(available_slots)
        bay, slot = self.get_bay_slot_from_index(selected_slot)

        # Mark slot as occupied
        self.mark_slot_occupied(selected_slot)

        # Calculate a random score between 50-150
        allocation_score = float(random.uniform(50, 150))

        return {
            "status": "Allocated",
            "bay_assigned": bay,
            "slot_assigned": slot,
            "allocation_score": allocation_score,
            "allocation_time": datetime.now().isoformat(),
        }


class AIAllocationStrategy(AllocationStrategy):
    """AI-based allocation using the trained model"""

    def __init__(self, smart_parking_system: SmartParkingSystem):
        super().__init__(
            smart_parking_system.env.num_bays, smart_parking_system.env.slots_per_bay
        )
        self.smart_parking = smart_parking_system

    def reset_environment(self):
        """Reset both local tracking and smart parking environment"""
        super().reset_environment()
        if hasattr(self, "smart_parking"):
            self.smart_parking.env.reset(occupancy_rate=0.0)

    def mark_slot_occupied(self, slot_index: int):
        """Mark slot as occupied in both local tracking and smart parking environment"""
        super().mark_slot_occupied(slot_index)
        # Also update the smart parking environment
        bay_idx = slot_index // self.slots_per_bay
        slot_idx = slot_index % self.slots_per_bay
        self.smart_parking.env.state[slot_index] = 1
        self.smart_parking.env.parking_lot[bay_idx, slot_idx] = 1

    def allocate_slot(self, vehicle_data: Dict) -> Dict:
        available_slots = self.get_available_slots()

        if not available_slots:
            return {
                "status": "Parking full",
                "allocation_score": 0.0,
                "allocation_time": datetime.now().isoformat(),
            }

        # Use the smart parking system for allocation
        allocation_result = self.smart_parking.allocate_parking(vehicle_data)

        if allocation_result["status"] != "Allocated":
            return allocation_result

        # Convert to 0-based index for tracking
        bay = allocation_result["bay_assigned"]
        slot = allocation_result["slot_assigned"]
        slot_index = (bay - 1) * self.slots_per_bay + (slot - 1)

        # Mark slot as occupied (already done in smart_parking.allocate_parking)
        # but we need to update our local tracking
        self.occupancy_map[slot_index] = 1

        return allocation_result


class SimulationManager:
    """Manages parking allocation simulations with different strategies"""

    def __init__(self, smart_parking_system: SmartParkingSystem = None):
        self.smart_parking = smart_parking_system

    def get_strategy(self, strategy_name: str, **kwargs) -> AllocationStrategy:
        """Factory method to get the appropriate allocation strategy"""
        if strategy_name == "sequential":
            return SequentialAllocationStrategy(**kwargs)
        elif strategy_name == "random":
            return RandomAllocationStrategy(**kwargs)
        elif strategy_name == "algorithm":
            if not self.smart_parking:
                raise ValueError("Smart parking system required for AI strategy")
            return AIAllocationStrategy(self.smart_parking)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

    def run_simulation(self, vehicles: List[Dict], strategy_name: str) -> Dict:
        """Run a simulation with the specified strategy"""
        start_time = datetime.now()

        # Get the appropriate strategy
        strategy = self.get_strategy(strategy_name)

        allocation_results = []
        successful_allocations = 0
        total_scores = []

        for vehicle_data in vehicles:
            try:
                # Prepare vehicle data for allocation
                vehicle_input = self._prepare_vehicle_data(vehicle_data)

                # Allocate using the strategy
                allocation = strategy.allocate_slot(vehicle_input)

                if allocation["status"] == "Allocated":
                    successful_allocations += 1
                    total_scores.append(allocation["allocation_score"])

                    allocation_results.append(
                        {
                            "vehicle_plate_num": vehicle_data["vehicle_plate_num"],
                            "status": "success",
                            "bay_assigned": allocation["bay_assigned"],
                            "slot_assigned": allocation["slot_assigned"],
                            "allocation_score": allocation["allocation_score"],
                            "allocation_time": allocation["allocation_time"],
                        }
                    )
                else:
                    allocation_results.append(
                        {
                            "vehicle_plate_num": vehicle_data["vehicle_plate_num"],
                            "status": "failed",
                            "error_message": allocation["status"],
                        }
                    )

            except Exception as e:
                allocation_results.append(
                    {
                        "vehicle_plate_num": vehicle_data["vehicle_plate_num"],
                        "status": "error",
                        "error_message": str(e),
                    }
                )

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Calculate metrics
        total_vehicles = len(vehicles)
        failed_allocations = total_vehicles - successful_allocations
        success_rate = (
            (successful_allocations / total_vehicles * 100) if total_vehicles > 0 else 0
        )
        average_score = sum(total_scores) / len(total_scores) if total_scores else None

        return {
            "strategy": strategy_name,
            "total_vehicles": total_vehicles,
            "successful_allocations": successful_allocations,
            "failed_allocations": failed_allocations,
            "success_rate": round(success_rate, 2),
            "average_allocation_score": (
                round(average_score, 2) if average_score else None
            ),
            "total_processing_time": round(processing_time, 3),
            "allocation_results": allocation_results,
            "final_occupancy": strategy.occupancy_map.tolist(),
            "parking_lot_state": strategy.parking_lot.tolist(),
        }

    def _prepare_vehicle_data(self, vehicle_data: Dict) -> Dict:
        """Prepare vehicle data with calculated fields for AI strategy"""
        from utils.parking_system import parse_datetime

        prepared_data = vehicle_data.copy()

        # Calculate duration and time features if needed
        if "duration_hours" not in prepared_data:
            arrival_time = parse_datetime(vehicle_data["arrival_time"])
            departure_time = parse_datetime(vehicle_data["departure_time"])
            duration_hours = (departure_time - arrival_time).total_seconds() / 3600
            prepared_data["duration_hours"] = duration_hours

        if "day_of_week" not in prepared_data:
            arrival_time = parse_datetime(vehicle_data["arrival_time"])
            prepared_data["day_of_week"] = arrival_time.weekday()

        if "hour_of_day" not in prepared_data:
            arrival_time = parse_datetime(vehicle_data["arrival_time"])
            prepared_data["hour_of_day"] = arrival_time.hour

        return prepared_data
