import datetime
import pathlib

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)

# --- 1. CONFIGURATION & SETUP ---
IS_RESAMPLED = True
HOME_DIR = pathlib.Path.home()
DAISEE_ROOT = pathlib.Path(HOME_DIR) / "Downloads/DAiSEE"
DATA_ROOT = DAISEE_ROOT / "Data"
LABEL_FOLDER = DAISEE_ROOT / "Labels"
CSV_DATA_PATH = '../extract_engagement/facemesh_pose_2d_badan.csv'
RESULTS_DIR = pathlib.Path("results")

# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Setup MLflow
mlflow.set_tracking_uri(f"sqlite:///{HOME_DIR}/.mlflow/mlflow.db")
mlflow.set_experiment("DAiSEE-Engagement")

# Get current date for filenames
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RESULTS_FILENAME = RESULTS_DIR / f"results_ordinal_logreg_{current_time}.csv"
MODEL_FILENAME = RESULTS_DIR / f"model_ordinal_logreg_{current_time}.pkl"
REPORT_FILENAME = RESULTS_DIR / f"report_ordinal_logreg_{current_time}.txt"

# --- 2. HELPER CLASS: ORDINAL CLASSIFIER ---


class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    """
    Adapts any binary classifier (like Logistic Regression) to handle 
    ordinal data using the Frank-Hall (1-vs-Followers) decomposition method.
    Other name is Ordinal Binary Decomposition.
    """

    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator
        self.classifiers_ = {}
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))

        # For K classes, we train K-1 binary classifiers
        # Classifier i predicts: Is the value > class[i]?
        for i in range(len(self.classes_) - 1):
            threshold_class = self.classes_[i]
            # Create binary target: 1 if label > threshold, 0 otherwise
            binary_y = (y > threshold_class).astype(int)

            clf = clone(self.base_estimator)
            clf.fit(X, binary_y)
            self.classifiers_[i] = clf

        return self

    def predict(self, X):
        # Summing the binary predictions gives the ordinal rank
        # E.g., if >0 is True, >1 is True, >2 is False -> Sum is 2
        preds = np.zeros(X.shape[0])
        for i, clf in self.classifiers_.items():
            preds += clf.predict(X)

        # Map sum back to original classes (assuming 0,1,2,3 structure)
        return preds.astype(int)

    def get_params(self, deep=True):
        return {"base_estimator": self.base_estimator}


# --- 3. LOAD & PREPROCESS DATA ---
print("Loading data...")
df = pd.read_csv(CSV_DATA_PATH, header=0)

# Clean data
cols_to_drop = ['person_id', 'Boredom', 'Confusion', 'Frustration ']
df = df.drop(
    columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

if IS_RESAMPLED:
    df = resample(df, n_samples=200)

# Load Labels


def get_clip_ids(filename):
    return pd.read_csv(LABEL_FOLDER / filename, header=0)['ClipID'] \
        .astype(str).apply(lambda x: x.split(sep='.')[0]).unique()


trainlabels = get_clip_ids('TrainLabels.csv')
testlabels = get_clip_ids('TestLabels.csv')
validationlabels = get_clip_ids('ValidationLabels.csv')

# Split Data
train = df[df['clip_id'].astype(str).isin(trainlabels)]
test = df[df['clip_id'].astype(str).isin(testlabels)]
validation = df[df['clip_id'].astype(str).isin(validationlabels)]

# Prepare X and y
feature_drop = ['filename', 'clip_id', 'Engagement']

X_train = train.drop(columns=feature_drop, errors='ignore')
y_train = train['Engagement']

X_test = test.drop(columns=feature_drop, errors='ignore')
y_test = test['Engagement']

X_val = validation.drop(columns=feature_drop, errors='ignore')
y_val = validation['Engagement']

print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

# --- 4. MLFLOW TRAINING RUN ---
print(f"\nStarting MLflow run (Timestamp: {current_time})...")

with mlflow.start_run(run_name=f"Ordinal_LogReg_{current_time}"):

    # A. Setup Tags & Params
    base_lr = LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='lbfgs',
        # class_weight='balanced' # Optional: Uncomment if classes are imbalanced
    )

    mlflow.log_text(str(list(X_train.columns)), 'features.txt')
    mlflow.set_tag("model_type", "OrdinalLogisticRegression")
    mlflow.set_tag("implementation", "Frank-Hall_Decomposition")
    mlflow.set_tag("algo_type", "Ordinal")

    # Log base estimator params manually since we are wrapping it
    mlflow.log_params(
        {f"base_{k}": v for k, v in base_lr.get_params().items()})

    # B. Train Ordinal Model
    print("Training Ordinal Logistic Regression...")

    # Enable Autologging
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        log_models=True
    )

    # Initialize wrapper
    clf = OrdinalClassifier(base_estimator=base_lr)
    clf.fit(X_train, y_train)

    # C. Evaluate on TEST Set
    print("Evaluating on Test Set...")
    y_pred = clf.predict(X_test)

    # Standard Classification Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Ordinal Specific Metrics
    # MAE: Distance between truth and prediction (Pred 0 vs True 3 is worse than Pred 2 vs True 3)
    mae = mean_absolute_error(y_test, y_pred)
    # QWK: Quadratic Weighted Kappa (Industry standard for ordinal agreement)
    kappa = cohen_kappa_score(y_test, y_pred, weights='quadratic')

    class_report_str = classification_report(y_test, y_pred)

    # Log Metrics
    metrics_dict = {
        'test_accuracy_score': accuracy,
        'test_precision_score': precision,
        'test_recall_score': recall,
        'test_f1_score': f1,
        'test_mae_score': mae,           # Lower is better
        'test_kappa_quadratic_score': kappa  # Higher is better
    }
    mlflow.log_metrics(metrics_dict)

    mlflow.log_text(class_report_str, 'test_classification_report.txt')

    # --- GENERATE & LOG CONFUSION MATRIX ---
    print("Generating Confusion Matrix...")

    # Calculate CM
    cm = confusion_matrix(y_test, y_pred, labels=[
        0, 1, 2, 3], normalize='true')

    # Create Figure
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=[0, 1, 2, 3])
    disp.plot(cmap='Blues', ax=ax, values_format='.2f')

    ax.set_title("Confusion Matrix: \n(Rounded Predictions)")

    # Log Figure to MLflow
    mlflow.log_figure(fig, "confusion_matrix.png")

    # Close plot to free memory
    plt.close(fig)

    print("\n--- Test Results ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"MAE:       {mae:.4f} (Ordinal Metric)")
    print(f"Kappa (Q): {kappa:.4f} (Ordinal Metric)")
    print(f"F1 Score:  {f1:.4f}")

    print("\nDetailed Report:")
    print(class_report_str)

    # D. Save & Log Local Files
    with open(REPORT_FILENAME, "w") as f:
        f.write(f"Date: {current_time}\n")
        f.write(f"Ordinal Strategy: Frank-Hall (K-1 Binary Classifiers)\n")
        f.write(class_report_str)
    mlflow.log_artifact(str(REPORT_FILENAME))

    results_data = {
        'Date': [current_time],
        'Model': ['Ordinal Logistic Regression'],
        'PathOrName': [CSV_DATA_PATH],
        'Accuracy': [accuracy],
        'MAE': [mae],
        'Kappa_Quad': [kappa],
        'F1_Score': [f1],
        'Base_Model_Params': [str(base_lr.get_params())],
        'Classification_Report': [class_report_str]
    }
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(RESULTS_FILENAME, index=False)
    print(f"Metrics saved locally to {RESULTS_FILENAME}")
    mlflow.log_artifact(str(RESULTS_FILENAME))

    # Save Model
    # Note: mlflow.sklearn.log_model might treat this custom class as a generic python object
    # or pickle it successfully. We explicitly save via joblib as requested.
    joblib.dump(clf, MODEL_FILENAME)
    mlflow.log_artifact(str(MODEL_FILENAME))
    print(f"Model saved locally to {MODEL_FILENAME}")

    print(
        f"Run Complete. View results in MLflow (Run ID: {mlflow.active_run().info.run_id})")
