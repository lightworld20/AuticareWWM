from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

app = FastAPI(title="Mood Prediction API")

last_prediction = {"Index": None, "Label": None, "timestamp": None}
class_names = {1: 'Amusement', 2: 'Calm', 0: 'Stress'}

class PredictionInput(BaseModel):
    prediction: float

@app.post("/store_prediction/")
def store_prediction(pred_input: PredictionInput):
    prediction_index = int(round(pred_input.prediction))  # round for float predictions like 1.0
    label = class_names.get(prediction_index, "Unknown")  # safe lookup

    last_prediction["Index"] = prediction_index
    last_prediction["Label"] = label
    last_prediction["timestamp"] = datetime.now().isoformat()

    return {"status": "stored"}

@app.get("/last_prediction/")
def get_last_prediction():
    return last_prediction
