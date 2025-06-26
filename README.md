# Smart Parking API

## 🚗 Overview

The Smart Parking API is an intelligent parking management system that uses machine learning and reinforcement learning algorithms to optimize parking space allocation. The system provides multiple allocation strategies, real-time parking status monitoring, and comprehensive simulation capabilities for comparing different allocation approaches.

## ✨ Key Features

### 🤖 AI-Powered Allocation

- **Machine Learning Models**: Trained models for allocation score prediction, bay selection, and slot assignment
- **Q-Learning Algorithm**: Reinforcement learning for optimal parking decisions based on time-of-day and priority levels
- **Time-Aware Allocation**: Peak hours consideration (8-10am, 5-7pm) for dynamic decision making

### 🎯 Multiple Allocation Strategies

1. **Algorithm Strategy**: AI/ML-based intelligent allocation using trained models
2. **Sequential Strategy**: First-come-first-served allocation to available slots
3. **Random Strategy**: Random assignment to available parking slots

### 🚙 Vehicle Priority System

- **Priority Levels**: 0 (Lowest) to 3 (Highest)
- **Vehicle Types**: Car, Truck, Motorcycle
- **Plate Types**: Private, Public, Government (affects priority calculation)

### 📊 Real-Time Management

- **Live Parking Status**: Real-time occupancy tracking across 4 bays with 10 slots each
- **Allocation History**: Complete audit trail of all parking allocations
- **Time-Based Management**: Automatic slot release based on departure times

### 🔬 Simulation & Comparison

- **Multi-Strategy Simulation**: Compare all allocation strategies with the same vehicle dataset
- **Performance Metrics**: Success rates, allocation scores, processing times
- **Strategy Databases**: Separate tracking for each allocation strategy

## 🏗️ System Architecture

```
├── FastAPI Application (app.py)
├── AI Models (ai_models/)
│   ├── score_model.pkl      # Allocation score prediction
│   ├── bay_model.pkl        # Bay selection model
│   ├── slot_model.pkl       # Slot assignment model
│   └── q_table_time_aware.npy # Q-learning table
├── Database Layer (db.py)
├── Models (models/)
│   └── models.py            # Pydantic data models
└── Core Logic (utils/)
    ├── parking_system.py    # AI parking system
    └── simulation_strategies.py # Strategy implementations
```

## 🐳 Docker Setup

### Prerequisites

- Docker installed on your system
- Docker Compose (optional, for advanced setups)

### Build and Run with Docker

1. **Clone the repository:**

```bash
git clone <repository-url>
cd smart-parking-api
```

2. **Build the Docker image:**

```bash
docker build -t smart-parking-api .
```

3. **Run the container:**

```bash
docker run -d \
  --name smart-parking \
  -p 8000:8000 \
  -v $(pwd)/ai_models:/app/ai_models \
  -v $(pwd)/parking_db*.json:/app/ \
  smart-parking-api
```

4. **Access the API:**

- API Documentation: http://localhost:8000/docs
- API Base URL: http://localhost:8000
- Health Check: http://localhost:8000/

### Environment Variables (Optional)

```bash
# Example with custom settings
docker run -d \
  --name smart-parking \
  -p 8000:8000 \
  -e PYTHONUNBUFFERED=1 \
  -v $(pwd)/ai_models:/app/ai_models \
  smart-parking-api
```

## 🛠️ Local Development Setup

### Prerequisites

- Python 3.12 or higher
- pip package manager

### Installation Steps

1. **Clone and navigate to the project:**

```bash
git clone <repository-url>
cd smart-parking-api
```

2. **Create virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Ensure AI models are present:**

```bash
# The ai_models/ directory should contain:
# - score_model.pkl
# - bay_model.pkl
# - slot_model.pkl
# - q_table_time_aware.npy
```

5. **Run the application:**

```bash
fastapi run app.py --host 0.0.0.0 --port 8000
```

## 🔌 API Endpoints

### Core Parking Operations

#### Allocate Parking Slot

```http
POST /api/parking/allocate
Content-Type: application/json

{
  "vehicle_plate_num": "ABC1234",
  "vehicle_plate_type": 0,
  "vehicle_type": 0,
  "arrival_time": "2025-06-26T09:30:00",
  "departure_time": "2025-06-26T17:30:00",
  "priority_level": 1
}
```

#### Get Parking Status

```http
GET /api/parking/status
```

#### Get All Allocations

```http
GET /api/parking/allocations?active_only=true&vehicle_plate_num=ABC1234
```

### Simulation & Analysis

#### Run Strategy Simulation

```http
POST /api/parking/simulate
Content-Type: application/json

{
  "vehicles": [
    {
      "vehicle_plate_num": "ABC1234",
      "vehicle_plate_type": 0,
      "vehicle_type": 0,
      "arrival_time": "2025-06-26T09:30:00",
      "departure_time": "2025-06-26T17:30:00",
      "priority_level": 1
    }
  ],
  "allocation_strategy": "algorithm"
}
```

#### Compare All Strategies

```http
POST /api/parking/compare
Content-Type: application/json

[
  {
    "vehicle_plate_num": "ABC1234",
    "vehicle_plate_type": 0,
    "vehicle_type": 0,
    "arrival_time": "2025-06-26T09:30:00",
    "departure_time": "2025-06-26T17:30:00",
    "priority_level": 1
  }
]
```

### Management Operations

#### Update Allocation

```http
PUT /api/parking/allocation/{allocation_id}
Content-Type: application/json

{
  "departure_time": "2025-06-26T18:00:00"
}
```

#### End Allocation

```http
DELETE /api/parking/allocation/{allocation_id}
```

## 📊 Data Models

### Vehicle Data Format

| Field                | Type   | Values       | Description                            |
| -------------------- | ------ | ------------ | -------------------------------------- |
| `vehicle_plate_num`  | string | Any string   | Unique vehicle identifier              |
| `vehicle_plate_type` | int    | 0, 1, 2      | 0: Private, 1: Public, 2: Government   |
| `vehicle_type`       | int    | 0, 1, 2      | 0: Car, 1: Truck, 2: Motorcycle        |
| `arrival_time`       | string | ISO datetime | Vehicle arrival time                   |
| `departure_time`     | string | ISO datetime | Expected departure time                |
| `priority_level`     | int    | 0-3          | Priority level (0: Lowest, 3: Highest) |

### Allocation Strategies

- `"sequential"` - First available slot allocation
- `"random"` - Random slot assignment
- `"algorithm"` - AI-powered optimal allocation

## 🧠 Machine Learning Features

### Allocation Score Prediction

The system uses trained ML models to predict allocation scores based on:

- Vehicle characteristics (type, plate type)
- Temporal features (hour of day, day of week)
- Stay duration
- Priority level

### Q-Learning Integration

- **State Space**: Parking lot configuration (4 bays × 10 slots)
- **Action Space**: Available parking slots
- **Reward Function**: Priority-based with time and location factors
- **Q-Table Dimensions**: [4 priority levels, 24 hours, 40 parking slots]

## 📈 Performance Metrics

The system tracks and reports:

- **Success Rate**: Percentage of successful allocations
- **Average Allocation Score**: Quality metric for parking assignments
- **Processing Time**: Time taken for allocation decisions
- **Occupancy Rate**: Real-time parking lot utilization

## 🗃️ Database Structure

The system maintains separate JSON databases for:

- `parking_db.json` - Main allocation records
- `parking_db_algorithm.json` - AI strategy allocations
- `parking_db_sequential.json` - Sequential strategy allocations
- `parking_db_random.json` - Random strategy allocations

## 🔧 Configuration

### Parking Lot Configuration

- **Bays**: 4 bays (numbered 1-4)
- **Slots per Bay**: 10 slots (numbered 1-10)
- **Total Capacity**: 40 parking slots

### Time-Based Features

- **Peak Hours**: 8-10am, 5-7pm (higher occupancy simulation)
- **Near Peak**: 7am, 10am, 4pm, 7pm (medium occupancy)
- **Regular Hours**: All other times (lower occupancy)

## 🚀 Example Usage

### Quick Start with Docker

```bash
# Build and run
docker build -t smart-parking-api .
docker run -d -p 8000:8000 smart-parking-api

# Test the API
curl -X POST "http://localhost:8000/api/parking/allocate" \
  -H "Content-Type: application/json" \
  -d '{
    "vehicle_plate_num": "TEST123",
    "vehicle_plate_type": 0,
    "vehicle_type": 0,
    "arrival_time": "2025-06-26T09:30:00",
    "departure_time": "2025-06-26T17:30:00",
    "priority_level": 1
  }'
```

### Simulation Example

```bash
curl -X POST "http://localhost:8000/api/parking/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "vehicles": [
      {
        "vehicle_plate_num": "CAR001",
        "vehicle_plate_type": 0,
        "vehicle_type": 0,
        "arrival_time": "2025-06-26T08:30:00",
        "departure_time": "2025-06-26T12:30:00",
        "priority_level": 1
      }
    ],
    "allocation_strategy": "algorithm"
  }'
```

## 📝 API Documentation

Once running, visit http://localhost:8000/docs for interactive API documentation powered by Swagger UI.
