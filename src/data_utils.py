import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(file_path: str):
    """Loads CSV data from the provided path."""
    return pd.read_csv(file_path)

def preprocess_data(df: pd.DataFrame):
    """Generic preprocessing for churn datasets with feature scaling and label encoding."""
    df = df.dropna()  # Drop missing values
    target = 'churn' if 'churn' in df.columns else df.columns[-1]  # Assumes target is last column if not named
    
    # Encoding categorical features
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Feature Scaling
    scaler = StandardScaler()
    X = df.drop(columns=[target])
    y = df[target]
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, label_encoders, scaler
    return X_scaled, y, label_encoders, scaler
