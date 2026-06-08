import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_model():
    print("--- Starting ML Model Training Pipeline ---")
    
    # 1. Load the dataset
    data_path = 'student_exam_scores.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
        
    df = pd.read_csv(data_path)
    print(f"Dataset successfully loaded. Shape: {df.shape}")
    
    # Display basic info
    print("\nDataset Columns and Types:")
    print(df.dtypes)
    
    # Check for missing values and handle them
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"\nFound {missing_count} missing values. Imputing with column means...")
        df = df.fillna(df.mean(numeric_only=True))
    else:
        print("\nNo missing values found.")

    # 2. Split features and target
    # Features: hours_studied, sleep_hours, attendance_percent, previous_scores
    # Target: exam_score
    X = df.drop(columns=['student_id', 'exam_score'])
    y = df['exam_score']
    
    print("\nFeatures selected:")
    print(list(X.columns))
    
    # 3. Train-Test Split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    # 4. Feature Scaling (essential for Linear Regression, helpful for general standard workflow)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Train & Evaluate Models
    # Model A: Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_preds = lr_model.predict(X_test_scaled)
    
    lr_mae = mean_absolute_error(y_test, lr_preds)
    lr_mse = mean_squared_error(y_test, lr_preds)
    lr_r2 = r2_score(y_test, lr_preds)
    
    print("\n--- Model A: Linear Regression Performance ---")
    print(f"Mean Absolute Error (MAE): {lr_mae:.3f}")
    print(f"Mean Squared Error (MSE): {lr_mse:.3f}")
    print(f"R-squared (R2) Score: {lr_r2:.3f}")
    
    # Model B: Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_preds = rf_model.predict(X_test_scaled)
    
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_mse = mean_squared_error(y_test, rf_preds)
    rf_r2 = r2_score(y_test, rf_preds)
    
    print("\n--- Model B: Random Forest Regressor Performance ---")
    print(f"Mean Absolute Error (MAE): {rf_mae:.3f}")
    print(f"Mean Squared Error (MSE): {rf_mse:.3f}")
    print(f"R-squared (R2) Score: {rf_r2:.3f}")
    
    # Choose best model based on R2 Score
    if rf_r2 > lr_r2:
        best_model = rf_model
        best_model_name = "Random Forest Regressor"
        best_r2 = rf_r2
        best_mae = rf_mae
        best_mse = rf_mse
    else:
        best_model = lr_model
        best_model_name = "Linear Regression"
        best_r2 = lr_r2
        best_mae = lr_mae
        best_mse = lr_mse
        
    print(f"\nSelecting {best_model_name} as the final model with R2 score: {best_r2:.3f}")
    
    # Save Feature Importance (if Random Forest is chosen)
    if best_model_name == "Random Forest Regressor":
        importances = rf_model.feature_importances_
        for col, imp in zip(X.columns, importances):
            print(f"Feature '{col}' Importance: {imp:.3f}")
            
    # 6. Save Model and Scaler
    print("\nSaving model assets...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    # Also save metadata for UI display
    metadata = {
        'model_name': best_model_name,
        'r2_score': float(best_r2),
        'mae': float(best_mae),
        'mse': float(best_mse),
        'features': list(X.columns)
    }
    with open('model_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
        
    print("Model assets and metadata successfully saved!")
    print("--- Pipeline Completed Successfully ---")

if __name__ == '__main__':
    train_model()
