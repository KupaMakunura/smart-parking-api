from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class VehicleData(BaseModel):
    vehicle_plate_num: str
    vehicle_plate_type: int = Field(..., description="0: Private, 1: Public, 2: Govt")
    vehicle_type: int = Field(..., description="0: Car, 1: Truck, 2: Motorcycle")
    arrival_time: str = Field(..., description="ISO format datetime string")
    departure_time: str = Field(..., description="ISO format datetime string")
    priority_level: int = Field(0, description="0-3, with 3 being highest priority")


class ParkingAllocation(BaseModel):
    id: Optional[int] = None
    vehicle_plate_num: str
    vehicle_plate_type: int
    vehicle_type: int
    bay_assigned: int
    slot_assigned: int
    allocation_score: float
    allocation_time: str
    departure_time: str
    priority_level: int
    is_active: bool = True


class SlotStatus(BaseModel):
    slot_number: int
    is_occupied: bool
    allocation: Optional[Dict[str, Any]] = None


class BayStatus(BaseModel):
    bay_number: int
    slots: List[SlotStatus]


class ParkingStatus(BaseModel):
    bays: List[BayStatus]
    total_slots: int
    occupied_slots: int
    available_slots: int
    occupancy_percentage: float
    updated_at: str
