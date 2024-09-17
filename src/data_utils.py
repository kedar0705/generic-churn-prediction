import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def preprocess_data(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])
    
    label_enc_cols = ['gender', 'hascrcard', 'isactivemember', 'churn'] 
    for col in label_enc_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    
    df = pd.get_dummies(df, columns=['geography'], drop_first=True)
    
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

if __name__ == "__main__":
    df = pd.read_csv("/home/user/Kedar/generic-churn-prediction/churn_data/Bank_churn.csv")
    df_preprocessed = preprocess_data(df)
    print(df_preprocessed.head())
    df_preprocessed = preprocess_data(df)
    print(df_preprocessed.head())
