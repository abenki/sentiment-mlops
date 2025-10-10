import argparse
from pathlib import Path
import json
from datetime import datetime

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from datasets import load_dataset
import joblib

from src.models.sentiment_model import SentimentModel


def load_data(sample_size=None):
    """Loads the IMDB dataset from Hugging Face"""
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb")
    
    train_data = pd.DataFrame(dataset['train'])
    test_data = pd.DataFrame(dataset['test'])
    
    if sample_size:
        print(f"Using a sample of {sample_size} samples for training")
        train_data = train_data.sample(n=sample_size, random_state=42)
    
    return train_data, test_data


def preprocess_text(text):
    """Basic text preprocessing"""
    # Remove HTML tags
    text = text.replace('<br />', ' ')
    text = text.replace('<br/>', ' ')
    return text


def compute_metrics(y_true, y_pred, y_pred_proba):
    """Calculates all metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba[:, 1])
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['true_negatives'] = int(cm[0, 0])
    metrics['false_positives'] = int(cm[0, 1])
    metrics['false_negatives'] = int(cm[1, 0])
    metrics['true_positives'] = int(cm[1, 1])
    
    return metrics


def train(
    max_features=5000,
    C=1.0,
    val_size=0.2,
    sample_size=None,
    experiment_name="sentiment-analysis",
    run_name=None
):
    """Main training function with MLflow tracking"""

    # MLflow configuration
    mlflow.set_experiment(experiment_name)

    # Start MLflow run
    with mlflow.start_run(run_name=run_name):
        
        # 1. DATA LOADING
        print("\n" + "="*80)
        print("DATA LOADING")
        print("="*80)
        
        train_data, test_data = load_data(sample_size=sample_size)
        
        # Log dataset info
        mlflow.log_param("dataset_name", "imdb")
        mlflow.log_param("train_size", len(train_data))
        mlflow.log_param("test_size", len(test_data))
        
        # 2. PREPROCESSING
        print("\n" + "="*80)
        print("PREPROCESSING")
        print("="*80)
        
        train_data['text_clean'] = train_data['text'].apply(preprocess_text)
        test_data['text_clean'] = test_data['text'].apply(preprocess_text)
        
        # Train/Val split
        X_train_full = train_data['text_clean'].values
        y_train_full = train_data['label'].values
        X_test = test_data['text_clean'].values
        y_test = test_data['label'].values
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=val_size,
            random_state=42,
            stratify=y_train_full
        )
        
        print(f"Train set: {len(X_train)} samples")
        print(f"Val set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Log params
        mlflow.log_param("val_size", val_size)
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("C", C)
        
        # 3. TRAINING
        print("\n" + "="*80)
        print("TRAINING")
        print("="*80)
        
        model = SentimentModel(max_features=max_features, C=C)

        print("Training the model...")
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()

        print(f"Training completed in {training_time:.2f}s")
        mlflow.log_metric("training_time_seconds", training_time)

        # 4. EVALUATION
        print("\n" + "="*80)
        print("EVALUATION")
        print("="*80)

        # Predictions
        print("\nEvaluating on validation set...")
        y_val_pred = model.predict(X_val)
        y_val_pred_proba = model.predict_proba(X_val)
        val_metrics = compute_metrics(y_val, y_val_pred, y_val_pred_proba)

        print("\nEvaluating on test set...")
        y_test_pred = model.predict(X_test)
        y_test_pred_proba = model.predict_proba(X_test)
        test_metrics = compute_metrics(y_test, y_test_pred, y_test_pred_proba)

        # Display results
        print("\n--- VALIDATION METRICS ---")
        for metric, value in val_metrics.items():
            if not metric.startswith('true_') and not metric.startswith('false_'):
                print(f"{metric}: {value:.4f}")
        
        print("\n--- TEST METRICS ---")
        for metric, value in test_metrics.items():
            if not metric.startswith('true_') and not metric.startswith('false_'):
                print(f"{metric}: {value:.4f}")

        # Log metrics in MLflow
        for metric, value in val_metrics.items():
            mlflow.log_metric(f"val_{metric}", value)
        
        for metric, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric}", value)
        
        # Classification report
        print("\n--- CLASSIFICATION REPORT (TEST) ---")
        print(classification_report(y_test, y_test_pred, target_names=['Negative', 'Positive']))

        # 5. MODEL SAVING AND LOGGING
        print("\n" + "="*80)
        print("MODEL SAVING AND LOGGING")
        print("="*80)

        # Create the models directory if it doesn't exist
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)

        # Save locally
        model_path = model_dir / "sentiment_model.joblib"
        joblib.dump(model, model_path)
        print(f"Model saved locally: {model_path}")

        # Create an input example for the signature
        input_example = X_test[:5]

        # Log the model in MLflow
        mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name="sentiment-classifier",
            input_example=input_example
        )
        print("Model registered in MLflow")

        # Log the vectorizer separately for inspection
        vectorizer_path = model_dir / "vectorizer.joblib"
        joblib.dump(model.vectorizer, vectorizer_path)
        mlflow.log_artifact(str(vectorizer_path))

        # Log prediction examples
        sample_predictions = pd.DataFrame({
            'text': X_test[:10],
            'true_label': y_test[:10],
            'predicted_label': y_test_pred[:10],
            'proba_negative': y_test_pred_proba[:10, 0],
            'proba_positive': y_test_pred_proba[:10, 1]
        })
        sample_pred_path = model_dir / "sample_predictions.csv"
        sample_predictions.to_csv(sample_pred_path, index=False)
        mlflow.log_artifact(str(sample_pred_path))

        # Log feature names (most important words)
        feature_names = model.vectorizer.get_feature_names_out()
        coefficients = model.classifier.coef_[0]

        # Top positive features
        top_positive_idx = np.argsort(coefficients)[-20:]
        top_positive = [(feature_names[i], coefficients[i]) for i in top_positive_idx]

        # Top negative features
        top_negative_idx = np.argsort(coefficients)[:20]
        top_negative = [(feature_names[i], coefficients[i]) for i in top_negative_idx]
        
        features_info = {
            'top_positive_features': top_positive,
            'top_negative_features': top_negative
        }
        
        features_path = model_dir / "important_features.json"
        with open(features_path, 'w') as f:
            json.dump(features_info, f, indent=2)
        mlflow.log_artifact(str(features_path))

        print("\nTraining completed successfully!")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        
        return model, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sentiment analysis model with MLflow")
    parser.add_argument("--max-features", type=int, default=5000, help="Max features for TF-IDF")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation set size")
    parser.add_argument("--sample-size", type=int, default=None, help="Sample size for quick training")
    parser.add_argument("--experiment-name", type=str, default="sentiment-analysis", help="MLflow experiment name")
    parser.add_argument("--run-name", type=str, default=None, help="MLflow run name")
    
    args = parser.parse_args()
    
    train(
        max_features=args.max_features,
        C=args.C,
        val_size=args.val_size,
        sample_size=args.sample_size,
        experiment_name=args.experiment_name,
        run_name=args.run_name
    )
