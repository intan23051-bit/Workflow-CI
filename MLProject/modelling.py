"""
Heart Disease Classification with MLflow
Author: Intan
Dataset: Heart Disease
"""

import setuptools  
import mlflow
import mlflow.sklearn
import os

mlflow.sklearn.autolog()

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings('ignore')

# MLflow Configuration 
# Set tracking URI based on environment
if os.getenv('GITHUB_ACTIONS'):
    # Running in GitHub Actions - use local file tracking
    mlflow.set_tracking_uri("file:./mlruns")
    print("Running in GitHub Actions - using local file tracking")
else:
    # Running locally - use tracking server or local files
    mlflow.set_tracking_uri("file:./mlruns")
    print("Running locally - using local file tracking")

# Set experiment
mlflow.set_experiment("heart-disease-classification")

def load_preprocessed_data():
    """Load preprocessed dataset"""
    print("\n" + "="*60)
    print("LOADING PREPROCESSED DATA")
    print("="*60)
    
    X_train = pd.read_csv('dataset_preprocessing/X_train_preprocessed.csv')
    X_test = pd.read_csv('dataset_preprocessing/X_test_preprocessed.csv')
    y_train = pd.read_csv('dataset_preprocessing/y_train_preprocessed.csv').values.ravel()
    y_test = pd.read_csv('dataset_preprocessing/y_test_preprocessed.csv').values.ravel()
    
    print(f"Data loaded successfully!")
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    print(f"   Features: {X_train.columns.tolist()}")
    
    # Handle any NaN or inf values
    for col in X_train.columns:
        if X_train[col].dtype in ['float64', 'int64']:
            X_train[col] = X_train[col].replace([np.inf, -np.inf], np.nan)
            X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan)
            
            if X_train[col].isnull().any():
                mean_val = X_train[col].mean()
                X_train[col] = X_train[col].fillna(mean_val)
                X_test[col] = X_test[col].fillna(mean_val)
                print(f"Filled NaN in '{col}' with mean ({mean_val:.2f})")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, X_test, y_test):
    """Train Random Forest model with MLflow autolog"""
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST MODEL")
    print("="*60)
    
    # Model configuration
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
    
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("="*60)
    
    # Log additional metrics manually
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    print("\nModel training completed successfully!")
    print(f"View results at: {mlflow.get_tracking_uri()}")
    
    return model, accuracy

def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("HEART DISEASE CLASSIFICATION - MODEL TRAINING")
    print("="*70)
    
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Train model (autolog will handle MLflow run automatically)
    model, accuracy = train_model(X_train, y_train, X_test, y_test)
    
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETED!")
    print("="*70)

if __name__ == "__main__":
    main()