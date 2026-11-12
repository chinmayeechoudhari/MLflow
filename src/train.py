"""
Assignment 2: Track Machine Learning Experiments Using MLflow
Course: MLOps
Topic: Experiment Tracking, Model Evaluation, Artifact Logging & Model Registration

This script implements an end-to-end Machine Learning pipeline with MLflow tracking:
1. Load dataset (Iris dataset from data/iris.csv)
2. Preprocess data (missing value handling, feature scaling, label encoding)
3. Split data (80% training, 20% testing)
4. Train at least 3 distinct models (Logistic Regression, Decision Tree, Random Forest)
5. Track and log parameters, metrics, confusion matrix, classification report, and trained models with MLflow
6. Compare experiment runs
7. Select the best performing model
8. Register the best model in the MLflow Model Registry and save to models/
"""

import os
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Define project base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "iris.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ARTIFACTS_TEMP_DIR = os.path.join(BASE_DIR, "temp_artifacts")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")
DB_PATH = os.path.join(BASE_DIR, "mlflow.db").replace(os.sep, "/")

# Allow file store for compatibility and use sqlite backend for robust Model Registry
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

EXPERIMENT_NAME = "Iris_Classification_Experiment"
REGISTERED_MODEL_NAME = "Iris_Best_Classifier"


def load_dataset(file_path: str) -> pd.DataFrame:
    """Step 1: Load dataset from CSV file."""
    print(f"[Step 1] Loading dataset from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully with shape: {df.shape}")
    print(f"Sample data:\n{df.head(3)}\n")
    return df


def preprocess_data(df: pd.DataFrame, target_col: str = "species"):
    """
    Step 2: Perform data preprocessing:
    - Handle missing values
    - Feature scaling using StandardScaler
    - Label encoding for categorical target
    """
    print("[Step 2] Performing preprocessing (missing values, feature scaling, label encoding)...")
    df_clean = df.copy()

    # 1. Handle missing values (if any)
    missing_counts = df_clean.isnull().sum()
    print(f"Missing values count per column:\n{missing_counts[missing_counts > 0] if missing_counts.sum() > 0 else 'No missing values found.'}")
    if missing_counts.sum() > 0:
        # Numeric columns filled with median
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        # Categorical columns filled with mode
        cat_cols = df_clean.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    # Separate features and target
    X_raw = df_clean.drop(columns=[target_col])
    y_raw = df_clean[target_col]
    feature_names = list(X_raw.columns)

    # 2. Label encoding for target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    target_names = list(label_encoder.classes_)
    print(f"Target classes encoded: {dict(enumerate(target_names))}")

    # 3. Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    return X_scaled, y_encoded, feature_names, target_names, scaler, label_encoder


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.20, random_state: int = 42):
    """Step 3: Split dataset into 80% training and 20% testing data."""
    print(f"[Step 3] Splitting data: {int((1-test_size)*100)}% Train, {int(test_size*100)}% Test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size:  {X_test.shape[0]} samples\n")
    return X_train, X_test, y_train, y_test


def save_and_log_confusion_matrix(y_true, y_pred, target_names, model_name, output_dir):
    """Helper to generate and save confusion matrix plot artifact."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=target_names,
        yticklabels=target_names
    )
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    cm_filepath = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(cm_filepath, dpi=300)
    plt.close()
    return cm_filepath


def save_and_log_classification_report(y_true, y_pred, target_names, model_name, output_dir):
    """Helper to generate and save classification report text artifact."""
    report_text = classification_report(y_true, y_pred, target_names=target_names)
    report_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)

    report_txt_path = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_classification_report.txt")
    with open(report_txt_path, "w") as f:
        f.write(f"Classification Report: {model_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(report_text)

    report_json_path = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_classification_report.json")
    with open(report_json_path, "w") as f:
        json.dump(report_dict, f, indent=4)

    return report_txt_path, report_json_path


def train_and_track_experiment(
    model_name: str,
    run_name: str,
    model,
    hyperparams: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: list,
    output_dir: str
):
    """
    Steps 4 & 5:
    - Train model
    - Track experiment with MLflow:
        Log: Parameters, Accuracy, Precision, Recall, F1 Score, Confusion Matrix, Classification Report, Trained Model
    """
    print(f"--- Running Experiment: {run_name} ({model_name}) ---")
    os.makedirs(output_dir, exist_ok=True)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        # Train model
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Compute evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        print(f"Results for {model_name}:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")

        # Log parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])
        for param_key, param_val in hyperparams.items():
            mlflow.log_param(param_key, param_val)

        # Log metrics (explicitly as requested in section 7 of assignment)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1 Score", f1)
        # Also log lowercase keys for programmatic comparison
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Generate and log Confusion Matrix artifact
        cm_path = save_and_log_confusion_matrix(y_test, y_pred, target_names, model_name, output_dir)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # Generate and log Classification Report artifact
        txt_path, json_path = save_and_log_classification_report(y_test, y_pred, target_names, model_name, output_dir)
        mlflow.log_artifact(txt_path, artifact_path="reports")
        mlflow.log_artifact(json_path, artifact_path="reports")

        # Log Trained Model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            pip_requirements=["scikit-learn>=1.3.0", "numpy>=1.24.0", "pandas>=2.0.0"],
            registered_model_name=None  # will register the best model in Step 8
        )

        print(f"Logged all parameters, metrics, artifacts, and model to MLflow for run {run_name}\n")
        return {
            "run_id": run_id,
            "run_name": run_name,
            "model_name": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "model": model
        }


def compare_all_experiments(experiment_id: str) -> pd.DataFrame:
    """Step 6: Compare all experiments recorded under the MLflow experiment."""
    print("[Step 6] Comparing all experiments recorded in MLflow...")
    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.f1_score DESC"]
    )

    comparison_data = []
    for r in runs:
        comparison_data.append({
            "Run ID": r.info.run_id,
            "Run Name": r.data.tags.get("mlflow.runName", "N/A"),
            "Model Name": r.data.params.get("model_name", "N/A"),
            "Accuracy": round(r.data.metrics.get("Accuracy", 0.0), 4),
            "Precision": round(r.data.metrics.get("Precision", 0.0), 4),
            "Recall": round(r.data.metrics.get("Recall", 0.0), 4),
            "F1 Score": round(r.data.metrics.get("F1 Score", 0.0), 4),
        })

    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPARISON TABLE")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80 + "\n")
    return comparison_df


def select_best_model(comparison_df: pd.DataFrame, trained_results: list):
    """Step 7: Select the best model based on highest F1 Score / Accuracy."""
    print("[Step 7] Selecting the best-performing model...")
    best_row = comparison_df.iloc[0]
    best_run_id = best_row["Run ID"]
    best_model_name = best_row["Model Name"]
    best_f1 = best_row["F1 Score"]
    best_acc = best_row["Accuracy"]

    print(f"Best Model Identified:")
    print(f"  Model:    {best_model_name}")
    print(f"  Run ID:   {best_run_id}")
    print(f"  Accuracy: {best_acc}")
    print(f"  F1 Score: {best_f1}\n")

    # Find the corresponding in-memory trained model object
    best_result = next((item for item in trained_results if item["run_id"] == best_run_id), None)
    return best_run_id, best_model_name, best_result


def register_best_model_in_mlflow(best_run_id: str, best_result: dict, registered_name: str = REGISTERED_MODEL_NAME):
    """
    Step 8: Register the best model in MLflow Model Registry and save locally to models/
    """
    print(f"[Step 8] Registering the best model '{registered_name}' in MLflow Model Registry...")
    model_uri = f"runs:/{best_run_id}/model"
    model_version = mlflow.register_model(model_uri=model_uri, name=registered_name)
    print(f"Model successfully registered!")
    print(f"  Model Name:    {model_version.name}")
    print(f"  Model Version: {model_version.version}")
    print(f"  Source URI:    {model_version.source}")

    # Also save the model artifact in the models/ folder as requested in guidelines
    os.makedirs(MODELS_DIR, exist_ok=True)
    local_model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    if best_result and "model" in best_result:
        joblib.dump(best_result["model"], local_model_path)
        print(f"Best model also saved locally to: {local_model_path}")

    return model_version


def main():
    print("=====================================================================")
    print("  ASSIGNMENT 2: TRACK MACHINE LEARNING EXPERIMENTS USING MLFLOW      ")
    print("=====================================================================\n")

    # MLflow Setup
    os.makedirs(MLRUNS_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_TEMP_DIR, exist_ok=True)
    tracking_uri = f"sqlite:///{DB_PATH}"
    mlflow.set_tracking_uri(tracking_uri)
    
    # Create or get experiment with artifact location pointing to mlruns
    client = MlflowClient()
    existing_exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if existing_exp is None:
        artifact_uri = Path(MLRUNS_DIR).resolve().as_uri()
        client.create_experiment(name=EXPERIMENT_NAME, artifact_location=artifact_uri)
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"MLflow Experiment:   {EXPERIMENT_NAME} (ID: {experiment.experiment_id})\n")

    # Step 1: Load Dataset
    df = load_dataset(DATA_PATH)

    # Step 2: Preprocess Data
    X_scaled, y_encoded, feature_names, target_names, scaler, encoder = preprocess_data(df, target_col="species")

    # Save preprocessing objects for production readiness
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(encoder, os.path.join(MODELS_DIR, "label_encoder.joblib"))

    # Step 3: Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y_encoded, test_size=0.20, random_state=42)

    # Step 4: Define at least 3 distinct models with different hyperparameters
    models_config = [
        {
            "model_name": "Logistic Regression",
            "run_name": "Logistic_Regression_Standard",
            "model": LogisticRegression(C=1.0, max_iter=200, solver="lbfgs", random_state=42),
            "hyperparams": {
                "C": 1.0,
                "max_iter": 200,
                "solver": "lbfgs",
                "random_state": 42
            }
        },
        {
            "model_name": "Decision Tree",
            "run_name": "Decision_Tree_Gini",
            "model": DecisionTreeClassifier(criterion="gini", max_depth=4, min_samples_split=2, random_state=42),
            "hyperparams": {
                "criterion": "gini",
                "max_depth": 4,
                "min_samples_split": 2,
                "random_state": 42
            }
        },
        {
            "model_name": "Random Forest",
            "run_name": "Random_Forest_Ensemble",
            "model": RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=2, random_state=42),
            "hyperparams": {
                "n_estimators": 100,
                "max_depth": 5,
                "min_samples_split": 2,
                "random_state": 42
            }
        }
    ]

    # Step 5: Train and Track experiments in MLflow
    trained_results = []
    for cfg in models_config:
        result = train_and_track_experiment(
            model_name=cfg["model_name"],
            run_name=cfg["run_name"],
            model=cfg["model"],
            hyperparams=cfg["hyperparams"],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            target_names=target_names,
            output_dir=ARTIFACTS_TEMP_DIR
        )
        trained_results.append(result)

    # Step 6: Compare all experiments
    comparison_df = compare_all_experiments(experiment.experiment_id)

    # Step 7: Select Best Model
    best_run_id, best_model_name, best_result = select_best_model(comparison_df, trained_results)

    # Step 8: Register Model in MLflow Model Registry
    model_version = register_best_model_in_mlflow(best_run_id, best_result, REGISTERED_MODEL_NAME)

    print("\n=====================================================================")
    print("MLflow Tracking & Registration Pipeline Completed Successfully!")
    print(f"Best Registered Model: {REGISTERED_MODEL_NAME} (Version: {model_version.version})")
    print("=====================================================================")


if __name__ == "__main__":
    main()
