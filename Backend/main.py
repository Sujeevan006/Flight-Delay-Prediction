import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import logging
from fastapi.middleware.cors import CORSMiddleware


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL_PATH = "flight_delay_model_xgb_tuned.joblib" 
SCALER_PATH = "standard_scaler_tuned.joblib"
ENCODER_PATH = "one_hot_encoder_tuned.joblib"
COLUMNS_PATH = "model_columns_tuned.joblib"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
    logger.info("Model and preprocessing artifacts loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model or preprocessing artifacts: {e}")
    raise


CATEGORICAL_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "DepTimeSlot"]


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)



class FlightData(BaseModel):
    Month: int
    DayofMonth: int
    DayOfWeek: int
    UniqueCarrier: str
    Origin: str
    Dest: str
    DepTime: int
    DepTimeSlot: str  


def create_time_slot(hour: int) -> str:
    if 6 <= hour < 10:
        return "Morning"
    elif 10 <= hour < 14:
        return "Midday"
    elif 14 <= hour < 18:
        return "Afternoon"
    elif 18 <= hour < 22:
        return "Evening"
    else:
        return "Night"


def preprocess_input(data: FlightData):
    try:
        df = pd.DataFrame([data.dict()])

        
        df["DepHour"] = df["DepTime"] // 100
        df["DepMinute"] = df["DepTime"] % 100

        df["DepTimeSlot"] = df["DepHour"].apply(create_time_slot)
        df.drop(["DepHour", "DepMinute", "DepTime"], axis=1, inplace=True)

        for col in CATEGORICAL_COLS:
            if col not in df.columns:
                raise ValueError(f"Missing column in input data: {col}")

        df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].astype(str)

        df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].fillna("Unknown")

        encoded_features = encoder.transform(df[CATEGORICAL_COLS])
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(CATEGORICAL_COLS))

 
        df.drop(columns=CATEGORICAL_COLS, inplace=True)


        df = pd.concat([df, encoded_df], axis=1)

        missing_cols = set(model_columns) - set(df.columns)
        for col in missing_cols:
            df[col] = 0  

        df = df[model_columns]


        logger.info(f"Processed DataFrame:\n{df.head()}")


        df_scaled = scaler.transform(df)

        return df_scaled
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {e}")


@app.post("/predict")
async def predict_delay(flight_data: FlightData):
    try:
        logger.info(f"Received prediction request: {flight_data.dict()}")


        processed_data = preprocess_input(flight_data)

        
        prediction_proba = model.predict_proba(processed_data)[:, 1]  
        delay_probability = float(prediction_proba[0])  

        
        logger.info(f"Prediction result: {delay_probability}")

        return {
            "prediction": "Delayed" if delay_probability > 0.5 else "On Time",
            "delay_probability": delay_probability
        }
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"message": "Flight Delay Prediction API is running!"}