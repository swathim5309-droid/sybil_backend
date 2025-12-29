from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import pickle
import os
import csv
import io
from tensorflow.keras.models import load_model

# -------------------------------
# App
# -------------------------------
app = FastAPI(title="Sybil Attack Detection API")

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
# Load models
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RF_PATH = os.path.join(BASE_DIR, "rf_model.pkl")
XGB_PATH = os.path.join(BASE_DIR, "xgb_model.pkl")
LSTM_PATH = os.path.join(BASE_DIR, "lstm_model.keras")
SS_MODEL_PATH = os.path.join(BASE_DIR, "sensor_model.pkl")

try:
    with open(RF_PATH, "rb") as f:
        rf_model = pickle.load(f)
    with open(XGB_PATH, "rb") as f:
        xgb_model = pickle.load(f)
    lstm_model = load_model(LSTM_PATH)
    with open(SS_MODEL_PATH, "rb") as f:
        ss_model = pickle.load(f)
    print("✅ Models loaded successfully")
except Exception as e:
    print("❌ Model load failed:", repr(e))
    rf_model = None
    xgb_model = None
    lstm_model = None
    ss_model = None

ACTION_MAP = {
    0: "No action required",
    1: "Alert driver",
    2: "Emergency brake",
}

# -------------------------------
# Health
# -------------------------------
@app.get("/")
def health():
    return {"status": "Backend running"}

# -------------------------------
# Ensemble prediction
# -------------------------------
def ensemble_predict(X_ml, X_dl):
    rf_pred = rf_model.predict(X_ml)
    xgb_pred = xgb_model.predict(X_ml)
    lstm_pred = lstm_model.predict(X_dl).argmax(axis=1)

    preds = np.vstack([rf_pred, xgb_pred, lstm_pred]).T
    final_pred = [np.bincount(row).argmax() for row in preds]
    return np.array(final_pred)

# -------------------------------
# Predict (manual / JSON)
# -------------------------------
@app.post("/predict")
def predict(data: InputData):
    if rf_model is None or xgb_model is None or lstm_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded")

    try:
        X_ml = np.array(data.features, dtype=float).reshape(1, -1)
        X_dl = np.array(data.features, dtype=float).reshape(1, -1)
        pred = ensemble_predict(X_ml, X_dl)[0]
        return {"prediction": int(pred)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# from fastapi import FastAPI, HTTPException, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List
# import numpy as np
# import pickle
# import os
# import csv
# import io

# # -------------------------------
# # App
# # -------------------------------
# app = FastAPI(title="Sybil Attack Detection API")


# # -------------------------------
# # CORS
# # -------------------------------
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=False,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# # model
# class InputData(BaseModel):
#     features: List[float] 
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     RF_PATH = os.path.join(BASE_DIR, "rf_model.pkl")
#     XGB_PATH = os.path.join(BASE_DIR, "xgb_model.pkl")
#     LSTM_PATH = os.path.join(BASE_DIR, "lstm_model.keras")
#     SS_MODEL_PATH = os.path.join(BASE_DIR, "sensor_model.pkl") 
#     # Load models 
#     try:
#         with open(RF_PATH, "rb") as f:
#             rf_model = pickle.load(f)
#         with open(XGB_PATH, "rb") as f:
#             xgb_model = pickle.load(f) lstm_model = load_model(LSTM_PATH)
#         with open(SS_MODEL_PATH, "rb") as f:
#             ss_model = pickle.load(f) print("✅ Models loaded successfully")
#     except Exception as e: 
#         print("❌ Model load failed:", repr(e))
#         ACTION_MAP = { 0: "No action required",
#                        1: "Alert driver", 
#                        2: "Emergency brake", }

# # -------------------------------
# # Input Schema (manual input)
# # -------------------------------
# # class InputData(BaseModel):
# #     features: List[float]

# # # -------------------------------
# # # Load Random Forest
# # # -------------------------------
# # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # MODEL_PATH = os.path.join(BASE_DIR, "ensemble_model (4).pkl")
# # SS_MODEL_PATH = os.path.join(BASE_DIR, "sensor_model.pkl")


# # model = None
# # try:
# #     with open(MODEL_PATH, "rb") as f:
# #         model = pickle.load(f)
# #     print("✅ Model loaded successfully:", type(model))
# #     with open(SS_MODEL_PATH, "rb") as f1:
# #         ss_model = pickle.load(f1)
# #     print("✅ Model loaded successfully:", type(model))
# # except Exception as e:
# #     print("❌ Model load failed:", repr(e))

# # print("MODEL PATH:", MODEL_PATH) 
# # print("MODEL EXISTS:", os.path.exists(MODEL_PATH))

# # -------------------------------
# # Health
# # -------------------------------
# @app.get("/")
# def health():
#     return {"status": "Backend running"}

# # -------------------------------
# # Predict (manual / JSON)
# # -------------------------------
# @app.post("/predict")
# def predict(data: InputData):
#     if model is None:
#         raise HTTPException(status_code=500, detail="Model not loaded")

#     try:
#         X = np.array(data.features, dtype=float).reshape(1, -1)
#         pred = model.predict(X)[0]

#         confidence = None
#         if hasattr(model, "predict_proba"):
#             confidence = float(max(model.predict_proba(X)[0]))

#         return {
#             "prediction": int(pred),
#             "confidence": confidence
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# REQUIRED_FEATURES = [
#    'x','y','speed','acceleration'
# ]

# @app.post("/predict-csv")
# async def predict_csv(file: UploadFile = File(...)):
#     if model is None:
#         raise HTTPException(status_code=500, detail="Model not loaded")

#     try:
#         content = await file.read()
#         decoded = content.decode("utf-8")
#         reader = csv.DictReader(io.StringIO(decoded))

#         # Read first data row
#         row = next(reader, None)
#         if row is None:
#             raise ValueError("CSV file is empty")

#         # Extract only required features in correct order
#         features = []
#         missing = []

#         for col in REQUIRED_FEATURES:
#             if col not in row:
#                 missing.append(col)
#             else:
#                 features.append(float(row[col]))

#         if missing:
#             raise ValueError(f"Missing required columns: {missing}")

#         X = np.array(features).reshape(1, -1)
#         pred = model.predict(X)[0]

#         confidence = None
#         if hasattr(model, "predict_proba"):
#             confidence = float(max(model.predict_proba(X)[0]))

#         return {
#             "prediction": int(pred),
#             "confidence": confidence,
#             "used_features": REQUIRED_FEATURES
#         }

#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))
        

SENSOR_REQUIRED_FEATURES = [
    "speed_kmh",
    "acceleration_mps2",
    "lane_deviation",
    "obstacle_distance",
    "traffic_density",
]

@app.post("/predict-sensor-csv")
async def predict_sensor_csv(file: UploadFile = File(...)):
    if ss_model is None:
        raise HTTPException(status_code=500, detail="Sensor model not loaded")

    try:
        # Read uploaded CSV
        content = await file.read()
        decoded = content.decode("utf-8")
        reader = csv.DictReader(io.StringIO(decoded))

        # Read first row only
        row = next(reader, None)
        if row is None:
            raise ValueError("CSV file is empty")

        features = []
        missing = []

        # Extract features in model order
        for col in SENSOR_REQUIRED_FEATURES:
            if col not in row:
                missing.append(col)
            else:
                try:
                    features.append(float(row[col]))
                except ValueError:
                    raise ValueError(
                        f"Invalid numeric value in column '{col}'"
                    )

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        # Run model
        X = np.array(features, dtype=float).reshape(1, -1)
        pred = int(ss_model.predict(X)[0])

        confidence = None
        if hasattr(ss_model, "predict_proba"):
            confidence = float(
                max(ss_model.predict_proba(X)[0])
            )

        action = ACTION_MAP.get(pred, "Unknown")

        return {
            "prediction": pred,
            "action": action,
            "confidence": confidence,
            "used_features": SENSOR_REQUIRED_FEATURES,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict-sensor-json")
def predict_sensor_json(data: InputData):
    if ss_model is None:
        raise HTTPException(status_code=500, detail="Sensor model not loaded")

    try:
        X = np.array(data.features, dtype=float).reshape(1, -1)
        pred = int(ss_model.predict(X)[0])

        confidence = None
        if hasattr(ss_model, "predict_proba"):
            confidence = float(max(ss_model.predict_proba(X)[0]))

        return {
            "prediction": pred,
            "action": ACTION_MAP.get(pred, "Unknown"),
            "confidence": confidence,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# previous -working
# -------------------------------
# Predict from CSV upload
# -------------------------------
# @app.post("/predict-csv")
# async def predict_csv(file: UploadFile = File(...)):
#     if model is None:
#         raise HTTPException(status_code=500, detail="Model not loaded")

#     try:
#         content = await file.read()
#         decoded = content.decode("utf-8")
#         reader = csv.reader(io.StringIO(decoded))

#         rows = list(reader)
#         if len(rows) < 2:
#             raise ValueError("CSV must have header + one data row")

#         values = [float(x) for x in rows[1]]
#         X = np.array(values).reshape(1, -1)

#         pred = model.predict(X)[0]

#         confidence = None
#         if hasattr(model, "predict_proba"):
#             confidence = float(max(model.predict_proba(X)[0]))

#         return {
#             "prediction": int(pred),
#             "confidence": confidence
#         }

#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

# # from fastapi import FastAPI
# # from pydantic import BaseModel
# # import pickle
# # import numpy as np
# # from fastapi.middleware.cors import CORSMiddleware
# # from typing import List

# # # --- Define EnsembleModel outside ---
# # class EnsembleModel:
# #     def __init__(self, rf_model, xgb_model, lstm_model):
# #         self.rf_model = rf_model
# #         self.xgb_model = xgb_model
# #         self.lstm_model = lstm_model

# #     def predict(self, X_ml, X_dl):
# #         rf_pred = self.rf_model.predict(X_ml)
# #         xgb_pred = self.xgb_model.predict(X_ml)
# #         lstm_pred = self.lstm_model.predict(X_dl).argmax(axis=1)

# #         preds = np.vstack([rf_pred, xgb_pred, lstm_pred]).T
# #         final_pred = [np.bincount(row).argmax() for row in preds]
# #         return np.array(final_pred)

# # # --- Load model ---
# # try:
# #     with open("/home/user/studio/backend/rf_model.pkl", "rb") as f:
# #         model = pickle.load(f)
# # except (FileNotFoundError, pickle.UnpicklingError) as e:
# #     print(f"ERROR: {e}")
# #     model = None

# # # --- FastAPI app ---
# # app = FastAPI()

# # # --- CORS ---
# # origins = ["http://localhost:3000"]
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],  # You can restrict this later
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # # --- Input schema ---
# # class InputData(BaseModel):
# #     features: List[float]

# # # --- Prediction endpoint ---
# # @app.post("/predict")
# # def predict(data: InputData):
# #     if model is None:
# #         return {"error": "Model not loaded. Please check backend logs."}
    
# #     try:
# #         features_array = np.array(data.features)

# #         # Adjust shapes based on your model requirements
# #         X_ml = features_array.reshape(1, -1)
# #         # Example for LSTM input (adjust as needed):
# #         # X_dl = features_array.reshape(1, 20, 8)

# #         prediction = model.predict(X_ml)  # Replace second arg with X_dl if needed
# #         return {"prediction": int(prediction[0])}
# #     except Exception as e:
# #         return {"error": f"Prediction failed: {str(e)}"}
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List
# import numpy as np
# import pickle
# import os

# # -------------------------------
# # FastAPI App
# # -------------------------------
# app = FastAPI(title="Sybil Attack Detection API")

# # -------------------------------
# # CORS (for React / Next.js)
# # -------------------------------
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # frontend
#     allow_credentials=False,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -------------------------------
# # Input Schema
# # -------------------------------
# class InputData(BaseModel):
#     features: List[float]

# # -------------------------------
# # Load Random Forest Model
# # -------------------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "rf_model.pkl")

# try:
#     with open(MODEL_PATH, "rb") as f:
#         model = pickle.load(f)
#     print("✅ Random Forest model loaded successfully")
# except Exception as e:
#     print("❌ Failed to load model:", e)
#     model = None

# # -------------------------------
# # Health Check
# # -------------------------------
# @app.get("/")
# def health_check():
#     return {"status": "Backend is running"}

# # -------------------------------
# # Prediction Endpoint
# # -------------------------------
# @app.post("/predict")
# def predict(data: InputData):
#     if model is None:
#         raise HTTPException(status_code=500, detail="Model not loaded")

#     try:
#         # Convert input to NumPy array
#         X = np.array(data.features, dtype=float).reshape(1, -1)

#         # Prediction
#         prediction = model.predict(X)[0]

#         # Optional confidence (if available)
#         confidence = None
#         if hasattr(model, "predict_proba"):
#             confidence = float(max(model.predict_proba(X)[0]))

#         return {
#             "prediction": int(prediction),
#             "confidence": confidence
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
# from fastapi import FastAPI, HTTPException, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List
# import numpy as np
# import pickle
# import os
# import csv
# import io

# # -------------------------------
# # App
# # -------------------------------
# app = FastAPI(title="Sybil Attack Detection API")


# # -------------------------------
# # CORS
# # -------------------------------
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=False,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # -------------------------------
# # Input Schema (manual input)
# # -------------------------------
# class InputData(BaseModel):
#     features: List[float]

# # -------------------------------
# # Load Random Forest
# # -------------------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "final_rf.pkl")


# model = None
# try:
#     with open(MODEL_PATH, "rb") as f:
#         model = pickle.load(f)
#     print("✅ Model loaded successfully:", type(model))
# except Exception as e:
#     print("❌ Model load failed:", repr(e))

# # print("MODEL PATH:", MODEL_PATH)
# # print("MODEL EXISTS:", os.path.exists(MODEL_PATH))

# # -------------------------------
# # Health
# # -------------------------------
# @app.get("/")
# def health():
#     return {"status": "Backend running"}

# # -------------------------------
# # Predict (manual / JSON)
# # -------------------------------
# @app.post("/predict")
# def predict(data: InputData):
#     if model is None:
#         raise HTTPException(status_code=500, detail="Model not loaded")

#     try:
#         X = np.array(data.features, dtype=float).reshape(1, -1)
#         pred = model.predict(X)[0]

#         confidence = None
#         if hasattr(model, "predict_proba"):
#             confidence = float(max(model.predict_proba(X)[0]))

#         return {
#             "prediction": int(pred),
#             "confidence": confidence
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# REQUIRED_FEATURES = [
#     "position_x",
#     "position_y",
#     "speed",
#     "direction",
#     "acceleration",
#     "signal_strength",
#     "trust_score",
#     "sybil_attack_attempts"
# ]

# @app.post("/predict-csv")
# async def predict_csv(file: UploadFile = File(...)):
#     if model is None:
#         raise HTTPException(status_code=500, detail="Model not loaded")

#     try:
#         content = await file.read()
#         decoded = content.decode("utf-8")
#         reader = csv.DictReader(io.StringIO(decoded))

#         # Read first data row
#         row = next(reader, None)
#         if row is None:
#             raise ValueError("CSV file is empty")

#         # Extract only required features in correct order
#         features = []
#         missing = []

#         for col in REQUIRED_FEATURES:
#             if col not in row:
#                 missing.append(col)
#             else:
#                 features.append(float(row[col]))

#         if missing:
#             raise ValueError(f"Missing required columns: {missing}")

#         X = np.array(features).reshape(1, -1)
#         pred = model.predict(X)[0]

#         confidence = None
#         if hasattr(model, "predict_proba"):
#             confidence = float(max(model.predict_proba(X)[0]))

#         return {
#             "prediction": int(pred),
#             "confidence": confidence,
#             "used_features": REQUIRED_FEATURES
#         }

#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))
        


    

# # previous -working
# # -------------------------------
# # Predict from CSV upload
# # -------------------------------
# # @app.post("/predict-csv")
# # async def predict_csv(file: UploadFile = File(...)):
# #     if model is None:
# #         raise HTTPException(status_code=500, detail="Model not loaded")

# #     try:
# #         content = await file.read()
# #         decoded = content.decode("utf-8")
# #         reader = csv.reader(io.StringIO(decoded))

# #         rows = list(reader)
# #         if len(rows) < 2:
# #             raise ValueError("CSV must have header + one data row")

# #         values = [float(x) for x in rows[1]]
# #         X = np.array(values).reshape(1, -1)

# #         pred = model.predict(X)[0]

# #         confidence = None
# #         if hasattr(model, "predict_proba"):
# #             confidence = float(max(model.predict_proba(X)[0]))

# #         return {
# #             "prediction": int(pred),
# #             "confidence": confidence
# #         }

# #     except Exception as e:
# #         raise HTTPException(status_code=400, detail=str(e))

# # # from fastapi import FastAPI
# # # from pydantic import BaseModel
# # # import pickle
# # # import numpy as np
# # # from fastapi.middleware.cors import CORSMiddleware
# # # from typing import List

# # # # --- Define EnsembleModel outside ---
# # # class EnsembleModel:
# # #     def __init__(self, rf_model, xgb_model, lstm_model):
# # #         self.rf_model = rf_model
# # #         self.xgb_model = xgb_model
# # #         self.lstm_model = lstm_model

# # #     def predict(self, X_ml, X_dl):
# # #         rf_pred = self.rf_model.predict(X_ml)
# # #         xgb_pred = self.xgb_model.predict(X_ml)
# # #         lstm_pred = self.lstm_model.predict(X_dl).argmax(axis=1)

# # #         preds = np.vstack([rf_pred, xgb_pred, lstm_pred]).T
# # #         final_pred = [np.bincount(row).argmax() for row in preds]
# # #         return np.array(final_pred)

# # # # --- Load model ---
# # # try:
# # #     with open("/home/user/studio/backend/rf_model.pkl", "rb") as f:
# # #         model = pickle.load(f)
# # # except (FileNotFoundError, pickle.UnpicklingError) as e:
# # #     print(f"ERROR: {e}")
# # #     model = None

# # # # --- FastAPI app ---
# # # app = FastAPI()

# # # # --- CORS ---
# # # origins = ["http://localhost:3000"]
# # # app.add_middleware(
# # #     CORSMiddleware,
# # #     allow_origins=["*"],  # You can restrict this later
# # #     allow_credentials=True,
# # #     allow_methods=["*"],
# # #     allow_headers=["*"],
# # # )

# # # # --- Input schema ---
# # # class InputData(BaseModel):
# # #     features: List[float]

# # # # --- Prediction endpoint ---
# # # @app.post("/predict")
# # # def predict(data: InputData):
# # #     if model is None:
# # #         return {"error": "Model not loaded. Please check backend logs."}
    
# # #     try:
# # #         features_array = np.array(data.features)

# # #         # Adjust shapes based on your model requirements
# # #         X_ml = features_array.reshape(1, -1)
# # #         # Example for LSTM input (adjust as needed):
# # #         # X_dl = features_array.reshape(1, 20, 8)

# # #         prediction = model.predict(X_ml)  # Replace second arg with X_dl if needed
# # #         return {"prediction": int(prediction[0])}
# # #     except Exception as e:
# # #         return {"error": f"Prediction failed: {str(e)}"}
# # from fastapi import FastAPI, HTTPException
# # from fastapi.middleware.cors import CORSMiddleware
# # from pydantic import BaseModel
# # from typing import List
# # import numpy as np
# # import pickle
# # import os

# # # -------------------------------
# # # FastAPI App
# # # -------------------------------
# # app = FastAPI(title="Sybil Attack Detection API")

# # # -------------------------------
# # # CORS (for React / Next.js)
# # # -------------------------------
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],  # frontend
# #     allow_credentials=False,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # # -------------------------------
# # # Input Schema
# # # -------------------------------
# # class InputData(BaseModel):
# #     features: List[float]

# # # -------------------------------
# # # Load Random Forest Model
# # # -------------------------------
# # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # MODEL_PATH = os.path.join(BASE_DIR, "rf_model.pkl")

# # try:
# #     with open(MODEL_PATH, "rb") as f:
# #         model = pickle.load(f)
# #     print("✅ Random Forest model loaded successfully")
# # except Exception as e:
# #     print("❌ Failed to load model:", e)
# #     model = None

# # # -------------------------------
# # # Health Check
# # # -------------------------------
# # @app.get("/")
# # def health_check():
# #     return {"status": "Backend is running"}

# # # -------------------------------
# # # Prediction Endpoint
# # # -------------------------------
# # @app.post("/predict")
# # def predict(data: InputData):
# #     if model is None:
# #         raise HTTPException(status_code=500, detail="Model not loaded")

# #     try:
# #         # Convert input to NumPy array
# #         X = np.array(data.features, dtype=float).reshape(1, -1)

# #         # Prediction
# #         prediction = model.predict(X)[0]

# #         # Optional confidence (if available)
# #         confidence = None
# #         if hasattr(model, "predict_proba"):
# #             confidence = float(max(model.predict_proba(X)[0]))

# #         return {
# #             "prediction": int(prediction),
# #             "confidence": confidence
# #         }

# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=str(e))
