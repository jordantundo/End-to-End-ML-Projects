
### Data Preprocessing (src/data_preprocessing.py)

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import yaml

def load_config():
    with open('config/config.yml', 'r') as f:
        return yaml.safe_load(f)

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df, config):
    # Separate features and target
    X = df.drop(columns=[config['target']])
    y = df[config['target']]
    
    # Get numerical and categorical features
    numerical_features = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
    categorical_features = [col for col in X.columns if X[col].dtype == 'object']
    
    # Create transformers
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Apply transformations
    X_processed = preprocessor.fit_transform(X)
    
    return X_processed, y, preprocessor

if __name__ == "__main__":
    config = load_config()
    df = load_data(config['data_path'])
    X, y, preprocessor = preprocess_data(df, config)
