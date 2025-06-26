from tinydb import TinyDB, Query
import json
from typing import Dict, Any


class ParkingDatabase:
    def __init__(self, db_path="./parking_db.json", encoder_class=None):
        self.db = TinyDB(db_path, encoding="utf-8", ensure_ascii=False)
        self.allocations = self.db.table("allocations")
        self.encoder_class = encoder_class
        self.VehicleQuery = Query()

    def create_allocation(self, allocation_data):
        # Convert NumPy types to Python native types if needed
        if self.encoder_class:
            allocation_data = json.loads(
                json.dumps(allocation_data, cls=self.encoder_class)
            )
        doc_id = self.allocations.insert(allocation_data)
        return doc_id

    def get_allocation(self, allocation_id: int):
        """Get allocation by ID"""
        result = self.allocations.get(doc_id=allocation_id)
        if result:
            result["id"] = allocation_id
        return result

    def update_allocation(self, allocation_id: int, update_data: Dict[str, Any]):
        """Update an existing allocation"""
        self.allocations.update(update_data, doc_ids=[allocation_id])
        updated_allocation = self.get_allocation(allocation_id)
        return updated_allocation

    def delete_allocation(self, allocation_id: int):
        """Delete an allocation (hard delete)"""
        self.allocations.remove(doc_ids=[allocation_id])
        return True

    def get_all_allocations(self):
        """Get all allocations"""
        results = self.allocations.all()
        for i, item in enumerate(results):
            item["id"] = self.allocations.all()[i].doc_id
        return results

    def get_filtered_allocations(self, filters: Dict[str, Any]):
        """Get allocations with filters"""
        query = None

        # Build query dynamically based on filters
        for key, value in filters.items():
            if query is None:
                query = getattr(self.VehicleQuery, key) == value
            else:
                query = query & (getattr(self.VehicleQuery, key) == value)

        if query is None:
            results = self.allocations.all()
        else:
            results = self.allocations.search(query)

        # Add document IDs as 'id' field
        for i, item in enumerate(results):
            if query is None:
                item["id"] = self.allocations.all()[i].doc_id
            else:
                item["id"] = self.allocations.search(query)[i].doc_id

        return results

    def clear_expired_allocations(self):
        """Mark allocations as inactive if they're past their departure time"""
        from datetime import datetime

        current_time = datetime.now().isoformat()
        query = (self.VehicleQuery.is_active == True) & (
            self.VehicleQuery.departure_time < current_time
        )

        expired_allocations = self.allocations.search(query)
        for allocation in expired_allocations:
            self.allocations.update({"is_active": False}, doc_ids=[allocation.doc_id])

        return len(expired_allocations)
