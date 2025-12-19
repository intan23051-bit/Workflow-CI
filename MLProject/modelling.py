"""
Heart Disease Classification with MLflow
Author: Intan
Dataset: Heart Disease
"""

import mlflow
import mlflow.sklearn
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import argparse

warnings.filterwarnings('ignore')

# MLflow Configuration 
if os.getenv('GITHUB_ACTIONS'):
    mlflow.set_tracking_uri("file:./mlruns")
    print("Running in GitHub Actions - using local file tracking")
else:
    mlflow.set_tracking_uri("file:./mlruns")
    print("Running locally - using local file tracking")

# Set experiment
mlflow.set_experiment("heart-disease-classification")

def load_preprocessed_data():
    """Load preprocessed dataset"""
    print("\n" + "="*60)
    print("LOADING PREPROCESSED DATA")
    print("="*60)
    
    # Memastikan path file sesuai dengan struktur folder Anda
    X_train = pd.read_csv('dataset_preprocessing/X_train_preprocessed.csv')
    X_test = pd.read_csv('dataset_preprocessing/X_test_preprocessed.csv')
    y_train = pd.read_csv('dataset_preprocessing/y_train_preprocessed.csv').values.ravel()
    y_test = pd.read_csv('dataset_preprocessing/y_test_preprocessed.csv').values.ravel()
    
    print(f"Data loaded successfully!")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, X_test, y_test, n_estimators, max_depth, min_samples_split, min_samples_leaf, random_state):
    """Train Random Forest model dengan parameter dari MLproject"""
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST MODEL")
    print("="*60)
    
    mlflow.sklearn.autolog()
    
    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state, # Sekarang sudah sinkron
            n_jobs=-1
        )
        
        print(f"Training with: n_estimators={n_estimators}, max_depth={max_depth}, random_state={random_state}")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        # Kalkulasi Metrik
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        
        # Log Metrik secara manual ke MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        print(f"Model Accuracy: {accuracy:.4f}")
        return model, accuracy

def main():
    """Main pipeline dengan Argparse untuk menangkap parameter dari MLproject"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=5)
    parser.add_argument("--min_samples_leaf", type=int, default=2)
    parser.add_argument("--random_state", type=int, default=42) 
    args = parser.parse_args()

    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # training
    train_model(
        X_train, y_train, X_test, y_test,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state
    )
    
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)

if __name__ == "__main__":
    main()