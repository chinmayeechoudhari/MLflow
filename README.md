# Assignment 2: Track Machine Learning Experiments Using MLflow

- **Course:** MLOps
- **Assignment Title:** Track Machine Learning Experiments Using MLflow
- **Difficulty Level:** Intermediate
- **Programming Language:** Python 3.10+
- **Frameworks:** MLflow, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

---

## 1. Aim
To develop a Machine Learning classification application and use **MLflow** to track experiments by logging hyperparameters, evaluation metrics, visual and textual artifacts, and trained models for end-to-end reproducibility, comparison, and automated model registration.

---

## 2. Problem Statement
Machine Learning development involves continuous experimentation with multiple algorithms, feature transformations, and hyperparameters. Managing these experiments manually or through ad-hoc logging is error-prone, untracked, and difficult to reproduce. 

This project provides an automated, production-grade MLOps pipeline using Python and MLflow that:
1. Records every experiment run automatically.
2. Logs hyperparameters, evaluation metrics, confusion matrix heatmaps, classification reports, and serialized model artifacts.
3. Compares multiple runs side-by-side to generate an experiment leaderboard.
4. Identifies the best-performing model based on evaluation metrics.
5. Registers the winning model into the **MLflow Model Registry** for versioning and deployment readiness.

---

## 3. Learning Objectives
After completing this assignment, you will be able to:
- Understand MLOps experiment tracking concepts and principles.
- Configure and run MLflow with local tracking and model registry backends.
- Train multiple classification algorithms (Logistic Regression, Decision Tree, Random Forest).
- Automatically log hyperparameters and evaluation metrics (Accuracy, Precision, Recall, F1 Score).
- Generate, store, and link evaluation artifacts (Confusion Matrix PNGs and Classification Reports).
- Query MLflow runs programmatically to compare experiment results.
- Register and version candidate models in the MLflow Model Registry.
- Apply MLOps industry best practices (version control, modular code, artifact isolation).

---

## 4. Project Directory Structure
```
MLflow_Assignment/
│
├── data/
│   └── iris.csv                 # Raw dataset (150 samples, 4 features, 3 target classes)
│
├── notebooks/
│   └── experiment.ipynb         # Interactive Jupyter Notebook with step-by-step documentation
│
├── src/
│   └── train.py                 # Modular, production-ready training and MLflow tracking script
│
├── models/
│   ├── best_model.joblib        # Serialized best-performing model artifact
│   ├── scaler.joblib            # Fitted StandardScaler for data inference
│   └── label_encoder.joblib     # Fitted LabelEncoder mapping classes
│
├── screenshots/
│   ├── 01_mlflow_experiments_overview.png
│   ├── 02_mlflow_model_registry.png
│   ├── 03_mlflow_run_details.png
│   ├── 04_mlflow_registered_model_details.png
│   ├── logistic_regression_confusion_matrix.png
│   ├── decision_tree_confusion_matrix.png
│   └── random_forest_confusion_matrix.png
│
├── mlruns/                      # MLflow local artifact repository for logged models and plots
├── mlflow.db                    # SQLite tracking backend for MLflow runs and Model Registry
├── requirements.txt             # Project dependencies and minimum versions
├── .gitignore                   # Version control ignore rules
└── README.md                    # Comprehensive documentation and experiment report
```

---

## 5. Software Requirements & Setup

### Prerequisites
- Python 3.10, 3.11, or above
- Git (for version control)
- A modern web browser (for viewing MLflow UI)

### Installation
Clone or navigate to the project directory and install dependencies:
```bash
pip install -r requirements.txt
```

`requirements.txt` includes:
```text
mlflow>=2.10.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
```

---

## 6. Functional Requirements & Implementation Details

The implementation strictly satisfies the 8 required functional steps:

### Step 1: Load Dataset
- The project loads the Iris dataset stored at `data/iris.csv`.
- Features: `sepal_length`, `sepal_width`, `petal_length`, `petal_width` (numerical).
- Target: `species` with 3 balanced classes (`setosa`, `versicolor`, `virginica`).

### Step 2: Data Preprocessing
- **Missing Value Handling:** Validates dataset completeness; handles any unexpected missing entries via median imputation for numerical features and mode for categorical features.
- **Label Encoding:** Encodes string targets into integer labels (`setosa`: 0, `versicolor`: 1, `virginica`: 2) using Scikit-learn's `LabelEncoder`.
- **Feature Scaling:** Standardizes features to zero mean and unit variance using `StandardScaler`.

### Step 3: Train-Test Split
- Dataset is partitioned into:
  - **Training Set (80%):** 120 samples
  - **Testing Set (20%):** 30 samples
- Uses stratified sampling (`stratify=y`) with a deterministic `random_state=42` to guarantee class balance across splits.

### Step 4: Train Multiple ML Models
At least three distinct classification models are trained:
1. **Logistic Regression:** Linear classifier optimized using the `lbfgs` solver (`C=1.0`, `max_iter=200`).
2. **Decision Tree Classifier:** Non-linear decision tree using Gini impurity criterion (`criterion='gini'`, `max_depth=4`, `min_samples_split=2`).
3. **Random Forest Classifier:** Bagging ensemble of 100 decision trees (`n_estimators=100`, `max_depth=5`, `min_samples_split=2`).

### Step 5: Track Every Experiment Using MLflow
For each model run, MLflow automatically captures:
- **Experiment Name:** `Iris_Classification_Experiment`
- **Run Name:** e.g., `Logistic_Regression_Standard`, `Decision_Tree_Gini`, `Random_Forest_Ensemble`
- **Parameters:** Algorithm hyperparameters (`C`, `max_iter`, `solver`, `criterion`, `max_depth`, `n_estimators`, `test_size`, `train_samples`, `test_samples`)
- **Metrics:**
  - `Accuracy`
  - `Precision` (weighted)
  - `Recall` (weighted)
  - `F1 Score` (weighted)
- **Artifacts Logged:**
  - `plots/<model_name>_confusion_matrix.png`: High-resolution heatmap of predictions vs. ground truth.
  - `reports/<model_name>_classification_report.txt`: Full precision, recall, and f1 per-class breakdown.
  - `reports/<model_name>_classification_report.json`: Machine-readable classification report.
- **Trained Model:** Serialized model logged using `mlflow.sklearn.log_model`.

### Step 6: Compare All Experiments
All runs under the experiment are queried programmatically through `MlflowClient.search_runs` and sorted by evaluation metrics.

### Step 7: Select Best-Performing Model
The system dynamically selects the model with the highest `F1 Score` and `Accuracy`.

### Step 8: Register Model in MLflow Model Registry
- The winning model is registered into the MLflow Model Registry under the registered name: **`Iris_Best_Classifier`**.
- Created as **Version 1** with full lineage linked back to its source run and training artifacts.
- The model is also exported locally to `models/best_model.joblib`.

---

## 7. Experimental Results & Performance Comparison

The following table summarizes the evaluation metrics recorded in MLflow across the test set (30 samples):

| Model Name | Run Name | Hyperparameters | Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | `Random_Forest_Ensemble` | `n_estimators=100, max_depth=5` | **0.9333** | **0.9333** | **0.9333** | **0.9333** |
| **Decision Tree** | `Decision_Tree_Gini` | `criterion=gini, max_depth=4` | **0.9333** | **0.9333** | **0.9333** | **0.9333** |
| **Logistic Regression** | `Logistic_Regression_Standard` | `C=1.0, solver=lbfgs, max_iter=200` | **0.9333** | **0.9333** | **0.9333** | **0.9333** |

### Detailed Classification Report (Best Model - Random Forest):
```
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       0.90      0.90      0.90        10
   virginica       0.90      0.90      0.90        10

    accuracy                           0.93        30
   macro avg       0.93      0.93      0.93        30
weighted avg       0.93      0.93      0.93        30
```

---

## 8. How to Execute

### Option A: Run the Automated Training & MLflow Tracking Script
```bash
python src/train.py
```
This executes all 8 steps sequentially, logs metrics and artifacts to MLflow, compares runs, and registers the winning model.

### Option B: Run via Jupyter Notebook
Launch Jupyter or open in VS Code:
```bash
jupyter notebook notebooks/experiment.ipynb
```
Follow the step-by-step cells with markdown explanations and interactive visualizations.

### Option C: Launch the MLflow Web UI
To inspect the experiments, metrics, comparison graphs, artifacts, and registered models:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Then navigate to `http://127.0.0.1:5000` in your web browser.

---

## 9. MLflow UI Verification & Artifacts
The `screenshots/` directory contains:
- `01_mlflow_experiments_overview.png`: Dashboard listing all experiment runs and comparative metrics.
- `02_mlflow_model_registry.png`: Model Registry table displaying `Iris_Best_Classifier`.
- `03_mlflow_run_details.png`: Run details page showcasing parameters, metrics, and logged artifact directories.
- `04_mlflow_registered_model_details.png`: Version details for the registered model.
- Confusion matrix artifacts for each model:
  - `logistic_regression_confusion_matrix.png`
  - `decision_tree_confusion_matrix.png`
  - `random_forest_confusion_matrix.png`

---

## 10. Conclusion
This assignment successfully demonstrated the full lifecycle of MLOps experiment tracking and model registration using **MLflow**:
- Multiple classification architectures were trained and systematically benchmarked under identical conditions.
- Every experiment run was tracked with full parameter and metric traceability.
- Confusion matrix visualizations and detailed reports were securely stored as MLflow artifacts.
- The best-performing model was programmatically identified and promoted to the **MLflow Model Registry**, fulfilling all functional and technical requirements.
