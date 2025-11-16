"""
Basic Model Training with MLflow Autolog
Author: Intan
Dataset: Heart Disease Classification
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

# MLflow Configuration
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("heart-disease-classification")

# Enable autolog
mlflow.sklearn.autolog()

def load_preprocessed_data():
    print("Loading preprocessed data")
    
    X_train = pd.read_csv('dataset_preprocessing/X_train_preprocessed.csv')
    X_test = pd.read_csv('dataset_preprocessing/X_test_preprocessed.csv')
    y_train = pd.read_csv('dataset_preprocessing/y_train_preprocessed.csv').values.ravel()
    y_test = pd.read_csv('dataset_preprocessing/y_test_preprocessed.csv').values.ravel()
    
    print(f"Data loaded successfully!")
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train Random Forest model"""
    print("\nTraining Random Forest model")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("Model trained successfully!")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\nEvaluating model")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC AUC   : {roc_auc:.4f}")
    print("="*50)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

def main():
    """Main training pipeline"""
    print("="*70)
    print("HEART DISEASE CLASSIFICATION - MODEL TRAINING")
    print("="*70)
    
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Start MLflow run
    with mlflow.start_run(run_name="random-forest-basic"):
        print("\nMLflow run started")
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        print("\nModel and metrics logged to MLflow")
        print(f"View results at: {mlflow.get_tracking_uri()}")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()