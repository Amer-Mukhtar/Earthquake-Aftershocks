# Earthquake Aftershock Prediction System - Complete Project Documentation

## 📋 Project Overview

This is an **end-to-end machine learning system** that predicts earthquake aftershocks using LSTM neural networks. Given a mainshock earthquake, the system predicts:
- **Magnitude** of aftershocks
- **Location** (latitude and longitude)
- **Estimated number** of aftershocks

The system consists of three main components:
1. **Data Processing Pipeline** - Identifies mainshock-aftershock pairs
2. **Model Training** - Trains a BiLSTM neural network
3. **Real-time API** - FastAPI service for live USGS earthquake data

---

## 🏗️ System Architecture

```
Input Earthquake Data (CSV)
         ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Part 1: DATA PROCESSING (process.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✓ Load & clean data
  ✓ Identify mainshock-aftershock pairs
  ✓ Create training datasets
  ✓ Normalize features
         ↓
   Processed Data (numpy arrays)
   └─ X_train.npy, X_test.npy
   └─ y_train.npy, y_test.npy
   └─ Scalers & metadata
         ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Part 2: MODEL TRAINING (train.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✓ Build BiLSTM neural network
  ✓ Train on mainshock→aftershock pairs
  ✓ Evaluate performance metrics
  ✓ Save trained model
         ↓
   Trained Model
   └─ aftershock_lstm_model.h5
   └─ Scalers & metadata
         ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Part 3: REAL-TIME PREDICTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✓ FastAPI service (earthquake_api.py)
  ✓ Fetch live USGS earthquake data
  ✓ Make predictions from trained model
  ✓ Interactive web interface
         ↓
   Aftershock Predictions
   └─ Magnitude, location, probability
```

---

## 📁 Project Files Explained

### 1. **process.py** - Data Processing Pipeline
**Purpose:** Converts raw earthquake CSV data into training datasets

**What it does:**
- **Loads** earthquake data from CSV (mainshocks + aftershocks)
- **Identifies pairs** using:
  - Time window: 30 days after mainshock
  - Distance threshold: 100 km from mainshock
  - Magnitude filter: aftershock < mainshock - 0.5
- **Creates training data:**
  - **Input (X):** Mainshock features [magnitude, depth, latitude, longitude]
  - **Output (y):** Aftershock features [magnitude, latitude, longitude]
- **Normalizes** using MinMaxScaler
- **Saves processed data** to `processed_data/` folder

**Key Functions:**
```python
load_data()                    # Load earthquake CSV
preprocess_data()             # Clean and convert data types
identify_mainshock_aftershock_pairs()  # Find M-A relationships
create_training_data()        # Prepare X, y arrays
normalize_data()              # Scale features 0-1
split_data()                  # 80% train, 20% test
save_processed_data()         # Save for model training
```

**Input:** `query (2).csv` - Raw earthquake data
**Output:** 
- `processed_data/X_train.npy` - Training input (mainshocks)
- `processed_data/y_train.npy` - Training output (aftershocks)
- `processed_data/X_test.npy` - Testing input
- `processed_data/y_test.npy` - Testing output
- Scalers & metadata for later use

---

### 2. **train.py** - Model Training & Prediction
**Purpose:** Trains LSTM model and makes predictions

**Part A: AftershockModelTrainer Class**
- **Loads** processed data from Part 1
- **Builds** BiLSTM neural network architecture:
  ```
  Input (4 features) 
    ↓
  Bidirectional LSTM (128 units)
    ↓
  Dropout (0.3)
    ↓
  Bidirectional LSTM (64 units)
    ↓
  Dropout (0.3)
    ↓
  Dense (32 units) + ReLU
    ↓
  Dropout (0.2)
    ↓
  Dense (16 units) + ReLU
    ↓
  Output (3 features: magnitude, lat, lon)
  ```
- **Trains** model with:
  - Optimizer: Adam
  - Loss: Mean Squared Error (MSE)
  - Callbacks: Early stopping, learning rate reduction
- **Evaluates** using R², RMSE, MAE metrics
- **Saves** model to `trained_model/`

**Part B: AftershockPredictor Class**
- **Loads** trained model and scalers
- **Takes** mainshock data as input
- **Estimates** number of aftershocks based on magnitude
- **Predicts** aftershock characteristics:
  - Magnitude (smaller than mainshock)
  - Location (within ~100 km)
  - Probability score
- **Ranks** by magnitude (largest first)
- **Visualizes** predictions on map

**Key Functions:**
```python
# Training
build_model()           # Create BiLSTM architecture
train(epochs=150)       # Train on processed data
evaluate()              # Test performance
plot_training_results() # Visualize metrics

# Prediction
get_mainshock_input()             # User input
estimate_num_aftershocks()        # M → number of aftershocks
predict_aftershock()              # Generate predictions
display_predictions()             # Show results
plot_predictions()                # Geographic visualization
```

**Input:** Mainshock data (magnitude, depth, latitude, longitude)
**Output:** 
- List of predicted aftershocks with:
  - Magnitude
  - Location (lat/lon)
  - Distance from mainshock
  - Probability score

---

### 3. **earthquake_api.py** - FastAPI Real-Time Service
**Purpose:** Provides REST API to fetch live earthquake data from USGS

**Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/earthquakes/recent` | GET | Get earthquakes from last N hours |
| `/earthquakes/by_time` | GET | Get earthquakes between two dates |

**Example Usage:**
```bash
# Start API
uvicorn earthquake_api:app --reload --port 8000

# Get recent earthquakes (M≥4.5, last 24 hours)
curl "http://127.0.0.1:8000/earthquakes/recent?hours=24&min_magnitude=4.5"

# Get earthquakes by date range
curl "http://127.0.0.1:8000/earthquakes/by_time?starttime=2025-01-01T00:00:00&endtime=2025-01-02T00:00:00"
```

**Data Sources:**
- USGS Earthquake Hazards Program (FDSN Web Services)
- URL: `https://earthquake.usgs.gov/fdsnws/event/1/query`

**Returns:** JSON with earthquake events containing:
- Event ID, time, magnitude
- Location (latitude, longitude, depth)
- USGS detail page URL

---

## 🚀 How to Use the System

### Step 1: Prepare Data
1. Get earthquake CSV file with columns: `time`, `mag`, `depth`, `latitude`, `longitude`
2. Place it in the project folder (e.g., `query (2).csv`)

### Step 2: Run Data Processing
```bash
python process.py
```
**What happens:**
- Loads your earthquake CSV
- Identifies ~50-100+ mainshock-aftershock pairs
- Cleans and normalizes data
- Saves to `processed_data/` folder
- Creates visualization: `aftershock_analysis.png`

**Time:** 2-5 minutes

### Step 3: Train Model
```bash
python train.py
```
**Select option: 1 (Train model)**

**What happens:**
- Loads processed data
- Builds and trains BiLSTM model
- Evaluates performance (R², RMSE, MAE)
- Saves model to `trained_model/`
- Creates visualization: `aftershock_model_results.png`

**Time:** 5-15 minutes (depends on data size)

### Step 4: Make Predictions
```bash
python train.py
```
**Select option: 2 (Predict aftershocks)**

**Interactive input:**
```
Enter mainshock data: 7.2, 15.0, 34.05, -118.25
```
Format: `magnitude, depth_km, latitude, longitude`

**Output:**
- List of predicted aftershocks
- Geographic map visualization
- Risk assessment (Low/Moderate/High/Critical)

### Step 5: Live Data API (Optional)
```bash
# Install API dependencies
pip install "fastapi[standard]" httpx uvicorn

# Start API server
uvicorn earthquake_api:app --reload --port 8000

# Open browser: http://127.0.0.1:8000/docs
```

---

## 📊 Data Flow Explanation

### Input Data Format (CSV)
```csv
time,mag,depth,latitude,longitude
2023-01-01T10:30:00Z,6.5,15.0,34.05,-118.25
2023-01-01T11:45:00Z,5.2,12.3,34.10,-118.20
2023-01-01T12:15:00Z,5.8,14.5,34.08,-118.22
...
```

### Mainshock-Aftershock Identification
```
MAINSHOCK: M6.5 at (34.05°N, 118.25°W)
├─ AFTERSHOCK 1: M5.2 at (34.10°N, 118.20°W) - 8.5 km away, 1.3 hours later
├─ AFTERSHOCK 2: M5.8 at (34.08°N, 118.22°W) - 5.2 km away, 1.8 hours later
└─ ...more aftershocks...
```

### Training Data Format
```
INPUT (X):  [6.5, 15.0, 34.05, -118.25]  ← mainshock
OUTPUT (y): [5.2, 34.10, -118.20]        ← aftershock (what model predicts)

INPUT (X):  [6.5, 15.0, 34.05, -118.25]
OUTPUT (y): [5.8, 34.08, -118.22]
```

### Model Prediction Process
```
MAINSHOCK INPUT: [6.5, 15.0, 34.05, -118.25]
         ↓
    [Normalize]
         ↓
    [BiLSTM layers]
         ↓
    [Dense layers]
         ↓
    Predicted output: [5.1, 34.09, -118.21]
         ↓
    [Denormalize]
         ↓
PREDICTED AFTERSHOCK: M5.1 at (34.09°N, 118.21°W)
```

---

## 🧠 Neural Network Architecture

### BiLSTM (Bidirectional LSTM)
Why LSTM?
- **Sequential data:** Earthquake patterns follow temporal relationships
- **Long-term dependencies:** Can capture patterns over time
- **Better than regular NN:** Remembers previous states

Why Bidirectional?
- Processes information forward AND backward
- Captures patterns in both directions
- Typically ~2% better accuracy

### Network Layers
```
Layer 1: Bidirectional LSTM (128 units)
  - Processes input through 256 LSTM cells total
  - Activation: ReLU
  - Returns sequences (pass to next LSTM layer)

Layer 2: Dropout (0.3)
  - Randomly drops 30% of neurons
  - Prevents overfitting

Layer 3: Bidirectional LSTM (64 units)
  - Processes from first layer
  - 128 LSTM cells total
  - Returns final state only

Layer 4: Dropout (0.2)

Layer 5: Dense (32 units)
  - Fully connected layer
  - 32 neurons, ReLU activation

Layer 6: Dropout (0.2)

Layer 7: Dense (16 units)
  - Further processing
  - 16 neurons, ReLU activation

Layer 8: Dense (3 units)
  - Output layer
  - Predicts: [magnitude, latitude, longitude]
  - No activation (linear regression)

Total Parameters: ~100,000+
```

### Training Process
- **Optimizer:** Adam (adaptive learning rate)
- **Loss:** Mean Squared Error (MSE)
- **Metrics:** Mean Absolute Error (MAE)
- **Callbacks:**
  - Early Stopping: Stop if validation loss doesn't improve for 20 epochs
  - Learning Rate Reduction: Cut learning rate by 50% if stuck
- **Epochs:** 150 (or until early stopping)
- **Batch Size:** 32 samples per update

---

## 📈 Performance Metrics Explanation

### R² Score (Coefficient of Determination)
- **Range:** 0 to 1 (higher is better)
- **Meaning:** 
  - 0.9+ = Excellent (explains 90%+ of variance)
  - 0.7-0.9 = Good
  - 0.5-0.7 = Moderate
  - <0.5 = Poor

### RMSE (Root Mean Squared Error)
- **Units:** Same as prediction (e.g., magnitude, degrees)
- **Meaning:** Average prediction error
- **Example:** RMSE=0.3 for magnitude → predictions off by ~0.3 on average

### MAE (Mean Absolute Error)
- **Units:** Same as prediction
- **Meaning:** Average absolute difference
- **More interpretable:** For magnitude, MAE=0.25 means ±0.25 average error

### Example Results:
```
Magnitude Prediction:
  R² = 0.82 → Model explains 82% of magnitude variation
  RMSE = 0.35 → Average error ±0.35 magnitude units
  MAE = 0.28 → Typical error ±0.28 magnitude units

Location Prediction:
  R² = 0.75 → Explains 75% of location variation
  MAE = 0.15° ≈ 16.7 km error
```

---

## 🔍 Understanding Key Concepts

### Mainshock-Aftershock Pair Criteria
1. **Temporal:**
   - Aftershock occurs 0-30 days AFTER mainshock
   - Why 30 days? Aftershock rate decays exponentially; most occur in first week

2. **Spatial:**
   - Aftershock within 100 km of mainshock epicenter
   - Why 100 km? Stress transfer from mainshock

3. **Magnitude:**
   - Aftershock magnitude < Mainshock - 0.5
   - Why? Smaller events in the aftermath (Gutenberg-Richter law)

### Haversine Distance
- Calculates distance between two points on Earth's surface
- Accounts for Earth's curvature
- Formula: `d = 2R * arcsin(sqrt(sin²(Δlat/2) + cos(lat₁) * cos(lat₂) * sin²(Δlon/2)))`
- R = Earth's radius = 6,371 km

### Feature Normalization
**Why normalize?**
- Neural networks train faster with data in 0-1 range
- Prevents features with large values from dominating

**MinMaxScaler Formula:**
```
X_normalized = (X - X_min) / (X_max - X_min)
X_normalized ∈ [0, 1]
```

**Example:**
```
Magnitude (range 3-7): M6.5 → 0.75
Latitude (range -90 to 90): 34.05° → 0.689
```

---

## 💾 Project Structure

```
Earthquake Aftershock/
├── process.py                    # Data processing pipeline
├── train.py                      # Model training & prediction
├── earthquake_api.py             # FastAPI service
├── query (2).csv                 # Input earthquake data
├── processed_data/               # Part 1 output
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   ├── y_test.npy
│   ├── scaler_input.pkl
│   ├── scaler_output.pkl
│   └── metadata.pkl
├── trained_model/                # Part 2 output
│   ├── aftershock_lstm_model.h5
│   ├── scaler_input.pkl
│   ├── scaler_output.pkl
│   └── metadata.pkl
├── aftershock_analysis.png       # Part 1 visualization
├── aftershock_model_results.png  # Part 2 visualization
└── aftershock_prediction.png     # Prediction visualization
```

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Data Processing | Pandas, NumPy | Data manipulation & numerical computing |
| Visualization | Matplotlib, Seaborn | Charts and plots |
| Scaling | Scikit-learn | Feature normalization |
| Deep Learning | TensorFlow, Keras | Neural network training |
| API | FastAPI | REST API server |
| HTTP Client | httpx | Async USGS API calls |

---

## ⚙️ Installation & Setup

### 1. Install Python Dependencies
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
pip install "fastapi[standard]" httpx uvicorn  # For API
```

### 2. Verify Installation
```bash
python -c "import tensorflow; print(f'TensorFlow version: {tensorflow.__version__}')"
```

### 3. Run the System
```bash
# Step 1: Process data
python process.py

# Step 2: Train model
python train.py
# Select option 1

# Step 3: Make predictions
python train.py
# Select option 2

# Step 4 (Optional): Start API
uvicorn earthquake_api:app --reload --port 8000
```

---

## 🐛 Troubleshooting

### "No mainshock-aftershock pairs found"
- **Cause:** CSV data doesn't contain M5+ earthquakes with nearby M4+ events
- **Solution:** Adjust `time_window_days`, `distance_km`, or `min_mainshock_mag` in `process.py`

### "Model not found" error
- **Cause:** Forgot to run training (step 2)
- **Solution:** Run `python train.py` and select option 1

### "TensorFlow/Keras version mismatch"
- **Cause:** Different TensorFlow versions between training and prediction
- **Solution:** Retrain model with current TensorFlow version

### Poor prediction accuracy
- **Cause:** Insufficient training data or poor data quality
- **Solution:**
  - Use larger earthquake dataset (100+ mainshocks)
  - Increase epochs or adjust learning rate
  - Try different network architecture

---

## 📚 References & Further Reading

1. **Earthquake Science:**
   - Gutenberg-Richter Law: log₁₀(N) = a - b*M
   - Omori's Law: Aftershock rate decays over time

2. **Machine Learning:**
   - LSTM Networks: Hochreiter & Schmidhuber (1997)
   - Bidirectional RNNs: Graves et al. (2005)

3. **Seismic Data:**
   - USGS Earthquake Hazards Program
   - FDSN Web Services Documentation

---

## ✅ Quick Start Checklist

- [ ] Have earthquake CSV file ready
- [ ] Install Python & dependencies
- [ ] Run `process.py` to prepare data
- [ ] Run `train.py` (option 1) to train model
- [ ] Run `train.py` (option 2) to make predictions
- [ ] (Optional) Run FastAPI for live data integration

---

## 📝 Summary

This project demonstrates a **complete ML pipeline**:
1. **Data collection & preparation** (identify patterns from raw data)
2. **Feature engineering** (extract meaningful features)
3. **Model training** (learn mainshock→aftershock relationships)
4. **Evaluation** (assess prediction accuracy)
5. **Deployment** (make real-time predictions via API)

The BiLSTM model learns complex patterns between mainshocks and aftershocks, enabling prediction of location and magnitude for future earthquakes!

