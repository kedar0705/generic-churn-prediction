# src/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
from data_utils import load_data, preprocess_data

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

@app.post("/predict/")
async def predict(req: PredictionRequest):
    # Load the dataset for preprocessing
    try:
        data_file = f'churn_data/{req.dataset_name}.csv'
        df = load_data(data_file)
        X, y, label_encoders, scaler = preprocess_data(df)
        
        # Load the corresponding model
        model = load_model(req.dataset_name)
        
        # Preprocess features from the user
        X_user = scaler.transform([req.features])
        
        # Predict churn
        prediction = model.predict(X_user)
        return {"prediction": int(prediction[0])}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Endpoint for switching datasets
@app.get("/datasets/")
async def list_datasets():
    return {"datasets": ["Bank_churn", "Telco-Customer-Churn", "E Commerce Dataset"]}
