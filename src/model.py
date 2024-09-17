import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model
# def load_model(model_path):
#     return joblib.load(model_path)

def predict(input_data, model):
    input_data = np.array(input_data).reshape(1, -1)  # Reshape for a single prediction
    return model.predict(input_data)

def train_model(X, y, model_name: str, model_type='random_forest'):
    """Train a machine learning model and save it."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == 'random_forest':
        model = RandomForestClassifier()
    elif model_type == 'logistic_regression':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
    else:
        raise ValueError("Unsupported model type")
    
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    
    # Ensure the models directory exists
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Save the model
    model_path = os.path.join(model_dir, f'{model_name}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return accuracy


from data_utils import load_data, preprocess_data
from model import train_model

# List of datasets and their names
datasets = ['Bank_churn', 'Telco-Customer-Churn', 'E Commerce Dataset']
for dataset in datasets:
    df = load_data(f'/home/user/Kedar/generic-churn-prediction/churn_data/{dataset}.csv')
    X, y, _, _ = preprocess_data(df)
    accuracy = train_model(X, y, model_name=dataset)
    print(f'Trained {dataset} model with accuracy: {accuracy}')
    