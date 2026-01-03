from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import pickle, joblib, os, csv, io

# -------------------------------
# App
# -------------------------------
app = FastAPI(title="Sybil & Sensor Detection API")

# -------------------------------
# CORS
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Input Schema
# -------------------------------
class InputData(BaseModel):
    features: List[float]

# -------------------------------
# Load Models
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SYBIL_MODEL_PATH = os.path.join(BASE_DIR, "final_rf.pkl")
SENSOR_MODEL_PATH = os.path.join(BASE_DIR, "sensor_model.pkl")

try:
    with open(SYBIL_MODEL_PATH, "rb") as f:
        sybil_model = pickle.load(f)
except Exception:
    sybil_model = None

try:
    sensor_model = joblib.load(SENSOR_MODEL_PATH)
except Exception:
    sensor_model = None

# -------------------------------
# Feature sets
# -------------------------------
SYBIL_REQUIRED_FEATURES = ["x", "y", "speed", "acceleration"]

SENSOR_REQUIRED_FEATURES = [
    "speed_kmh",
    "acceleration_mps2",
    "lane_deviation",
    "obstacle_distance",
    "traffic_density",
]

ACTION_MAP = {
    0: "Normal Driving",
    1: "Sensor Spoofing Detected",
    2: "Emergency Stop",
}

# -------------------------------
# Health
# -------------------------------
@app.get("/")
def health():
    return {"status": "Backend running"}

# -------------------------------
# Sybil Attack Detection
# -------------------------------
@app.post("/predict")
def predict_sybil(data: InputData):
    if sybil_model is None:
        raise HTTPException(status_code=500, detail="Sybil model not loaded")

    X = np.array(data.features, dtype=float).reshape(1, -1)
    pred = int(sybil_model.predict(X)[0])

    confidence = None
    if hasattr(sybil_model, "predict_proba"):
        confidence = float(max(sybil_model.predict_proba(X)[0]))

    return {"prediction": pred, "confidence": confidence}

@app.post("/predict-csv")
async def predict_sybil_csv(file: UploadFile = File(...)):
    if sybil_model is None:
        raise HTTPException(status_code=500, detail="Sybil model not loaded")

    content = await file.read()
    decoded = content.decode("utf-8")
    reader = csv.DictReader(io.StringIO(decoded))

    row = next(reader, None)
    if row is None:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    features, missing = [], []
    for col in SYBIL_REQUIRED_FEATURES:
        if col not in row:
            missing.append(col)
        else:
            features.append(float(row[col]))

    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")

    X = np.array(features).reshape(1, -1)
    pred = int(sybil_model.predict(X)[0])

    confidence = None
    if hasattr(sybil_model, "predict_proba"):
        confidence = float(max(sybil_model.predict_proba(X)[0]))

    return {"prediction": pred, "confidence": confidence, "used_features": SYBIL_REQUIRED_FEATURES}

# -------------------------------
# Sensor Spoofing Detection
# -------------------------------
@app.post("/predict-sensor-json")
def predict_sensor_json(data: InputData):
    if sensor_model is None:
        raise HTTPException(status_code=500, detail="Sensor model not loaded")

    X = np.array(data.features, dtype=float).reshape(1, -1)
    pred = int(sensor_model.predict(X)[0])

    confidence = None
    if hasattr(sensor_model, "predict_proba"):
        confidence = float(max(sensor_model.predict_proba(X)[0]))

    return {"prediction": pred, "action": ACTION_MAP.get(pred, "Unknown"), "confidence": confidence}

@app.post("/predict-sensor-csv")
async def predict_sensor_csv(file: UploadFile = File(...)):
    if sensor_model is None:
        raise HTTPException(status_code=500, detail="Sensor model not loaded")

    content = await file.read()
    decoded = content.decode("utf-8")
    reader = csv.DictReader(io.StringIO(decoded))

    row = next(reader, None)
    if row is None:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    features, missing = [], []
    for col in SENSOR_REQUIRED_FEATURES:
        if col not in row:
            missing.append(col)
        else:
            try:
                features.append(float(row[col]))
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid numeric value in column '{col}'")

    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")

    X = np.array(features, dtype=float).reshape(1, -1)
    pred = int(sensor_model.predict(X)[0])

    confidence = None
    if hasattr(sensor_model, "predict_proba"):
        confidence = float(max(sensor_model.predict_proba(X)[0]))

    return {
        "prediction": pred,
        "action": ACTION_MAP.get(pred, "Unknown"),
        "confidence": confidence,
        "used_features": SENSOR_REQUIRED_FEATURES,
    }
