import os
import pickle
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from data_utils import load_data, preprocess_data
from model import load_model, predict


app = FastAPI()

class PredictionRequest(BaseModel):
    dataset_name: str
    features: list

def load_model(model_name: str):
    """Load a pre-trained model from file."""
    model_path = f'models/{model_name}.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

@app.get("/")
def read_root():
    return {"message": "Welcome to the Churn Prediction API"}

@app.post("/predict/")
async def predict(req: PredictionRequest):
    """
    Predict churn based on a dataset name and feature inputs.
    """
    try:
        # Load the dataset for preprocessing
        data_file = f'churn_data/{req.dataset_name}.csv'
        df = load_data(data_file)
        
        # Preprocess the dataset to match model expectations
        X, y, label_encoders, scaler = preprocess_data(df)
        
        # Load the corresponding model based on the dataset name
        model = load_model(req.dataset_name)
        
        # Preprocess user features for prediction (scale to match model input)
        X_user = scaler.transform([req.features])
        
        # Predict churn using the model
        prediction = model.predict(X_user)
        
        # Return the prediction result
        return {"prediction": int(prediction[0])}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Endpoint for listing available datasets
@app.get("/datasets/")
async def list_datasets():
    return {"datasets": ["Bank_churn", "Telco-Customer-Churn", "E Commerce Dataset"]}
