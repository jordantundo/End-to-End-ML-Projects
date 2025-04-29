import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from data_preprocessing import load_data, preprocess_data, load_config

def train_model(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Initialize model
    model = RandomForestRegressor(random_state=42)
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    
    # Evaluate
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Test MSE: {mse:.2f}")
    print(f"Test R2 Score: {r2:.2f}")
    
    return best_model

if __name__ == "__main__":
    config = load_config()
    df = load_data(config['data_path'])
    X, y, preprocessor = preprocess_data(df, config)
    
    model = train_model(X, y)
    
    # Save model and preprocessor
    joblib.dump(model, 'models/model.pkl')
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
