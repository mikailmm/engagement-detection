import datetime
import pathlib
import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import resample

# Import Base Estimators
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    cohen_kappa_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# --- 1. CONFIGURATION & SETUP ---
IS_RESAMPLED = False  # Set to True to speed up debugging
RESAMPLE_SIZE = 1000
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

# Global timestamp for this batch
BATCH_TIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# --- 2. HELPER CLASS: ORDINAL CLASSIFIER ---


class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    """
    Adapts any binary classifier to handle ordinal data using 
    Frank-Hall (1-vs-Followers) decomposition.
    """

    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator
        self.classifiers_ = {}
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        # Train K-1 classifiers
        for i in range(len(self.classes_) - 1):
            threshold_class = self.classes_[i]
            # Binary target: 1 if > threshold, 0 otherwise
            binary_y = (y > threshold_class).astype(int)
            clf = clone(self.base_estimator)
            clf.fit(X, binary_y)
            self.classifiers_[i] = clf
        return self

    def predict(self, X):
        preds = np.zeros(X.shape[0])
        for i, clf in self.classifiers_.items():
            preds += clf.predict(X)
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
    print(f"RESAMPLING ENABLED: Reducing dataset to {RESAMPLE_SIZE} samples.")
    df = resample(df, n_samples=RESAMPLE_SIZE, random_state=42)


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

feature_drop = ['filename', 'clip_id', 'Engagement']

X_train = train.drop(columns=feature_drop, errors='ignore')
y_train = train['Engagement']

X_test = test.drop(columns=feature_drop, errors='ignore')
y_test = test['Engagement']

print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

# --- 4. DEFINE BASE MODELS ---
# We define the base estimators here. They will be wrapped in OrdinalClassifier later.
base_models_config = [
    (
        "LogisticRegression",
        LogisticRegression(random_state=42, max_iter=2000, solver='lbfgs')
    ),
    # (
    #     "RandomForest",
    #     RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    # ),
    (
        "ExtraTrees",
        ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    ),
    (
        "KNN",
        KNeighborsClassifier(n_neighbors=5)
    ),
    (
        "GaussianNB",
        GaussianNB()
    ),
    # (
    #     "DecisionTree",
    #     DecisionTreeClassifier(random_state=42)
    # ),
    # Heavy Models (Comment out if testing quickly)
    # (
    #     "GradientBoosting",
    #     GradientBoostingClassifier(
    #         n_estimators=100, learning_rate=0.1, random_state=42, verbose=1)
    # ),
    (
        "MLP_NeuralNet",
        MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42,
                      verbose=True)
    ),
    # (
    #     "SVM",
    #     SVC(kernel='rbf', probability=True, random_state=42)
    # ),
]

# List to store summary
all_results_summary = []

# --- 5. TRAINING FUNCTION ---


def train_and_log_ordinal(model_name, base_clf, X_train, y_train, X_test, y_test):
    # Construct the display name
    full_model_name = f"Ordinal_{model_name}"
    print(f"\n[{full_model_name}] Starting training...")

    results_filename = RESULTS_DIR / \
        f"results_{full_model_name}_{BATCH_TIME}.csv"
    model_filename = RESULTS_DIR / f"model_{full_model_name}_{BATCH_TIME}.pkl"
    report_filename = RESULTS_DIR / \
        f"report_{full_model_name}_{BATCH_TIME}.txt"

    with mlflow.start_run(run_name=f"{'resampled_' if IS_RESAMPLED else ''}{full_model_name}_{BATCH_TIME}"):

        # 1. Log Setup
        mlflow.log_text(str(list(X_train.columns)), 'features.txt')
        mlflow.set_tag("model_type", full_model_name)
        mlflow.set_tag("base_estimator", model_name)
        mlflow.set_tag("algo_type", "Ordinal")

        # Enable Autologging
        mlflow.sklearn.autolog(
            log_input_examples=True,
            log_model_signatures=True,
            log_models=True
        )

        # Log params of the base estimator explicitly
        mlflow.log_params(
            {f"base_{k}": v for k, v in base_clf.get_params().items()})

        # 2. Wrap & Train
        clf = OrdinalClassifier(base_estimator=base_clf)
        clf.fit(X_train, y_train)
        print(f"[{full_model_name}] Training complete. Evaluating...")

        # 3. Predict
        y_pred = clf.predict(X_test)

        # 4. Calculate Metrics
        # Standard
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Ordinal Specific
        mae = mean_absolute_error(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred, weights='quadratic')

        class_report_str = classification_report(y_test, y_pred)

        # 5. Log Metrics

        mlflow.log_metrics({
            'test_mae_score': mae,  # Lower is better
            'test_accuracy_score': accuracy,
            'test_precision_score': precision,
            'test_recall_score': recall,
            'test_f1_score': f1,
            'test_kappa_quadratic_score': kappa  # Higher is better
        })
        mlflow.log_text(class_report_str, 'test_classification_report.txt')

        # 6. Generate & Log Confusion Matrix
        print(f"[{full_model_name}] Generating Confusion Matrix...")
        cm = confusion_matrix(y_test, y_pred, labels=[
                              0, 1, 2, 3], normalize='true')

        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=[0, 1, 2, 3])
        disp.plot(cmap='Blues', ax=ax, values_format='.2f')
        ax.set_title(f"Confusion Matrix (Normalized)\n{full_model_name}")

        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        print(
            f"[{full_model_name}] Acc: {accuracy:.4f} | MAE: {mae:.4f} | Kappa: {kappa:.4f}")

        # 7. Save Artifacts Locally
        with open(report_filename, "w") as f:
            f.write(
                f"Date: {BATCH_TIME}\nModel: {full_model_name}\nType: Ordinal (Frank-Hall)\n")
            f.write(class_report_str)
        mlflow.log_artifact(str(report_filename))

        joblib.dump(clf, model_filename)
        # mlflow.log_artifact(str(model_filename)) # Optional upload of full model

        # 8. Return Summary Data
        results_data = {
            'Date': [BATCH_TIME],
            'Model': [full_model_name],
            'Base_Estimator': [model_name],
            'PathOrName': [CSV_DATA_PATH],
            'Accuracy': [accuracy],
            'MAE': [mae],
            'Kappa_Quad': [kappa],
            'F1_Score': [f1],
            'Classification_Report': [class_report_str]
        }
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(results_filename, index=False)
        mlflow.log_artifact(str(results_filename))

        return results_data


# --- 6. MAIN EXECUTION LOOP ---
print(
    f"--- Starting Batch Ordinal Training: {len(base_models_config)} Base Models ---")

for model_name, base_instance in base_models_config:
    try:
        # Train
        res = train_and_log_ordinal(
            model_name, base_instance, X_train, y_train, X_test, y_test)

        # Append to summary
        summary_row = {k: v[0] for k, v in res.items()}
        all_results_summary.append(summary_row)

    except Exception as e:
        print(f"!!! Error training {model_name}: {e}")
        # import traceback
        # traceback.print_exc()

# --- 7. SAVE MASTER SUMMARY ---
if all_results_summary:
    summary_filename = RESULTS_DIR / f"SUMMARY_ORDINAL_{BATCH_TIME}.csv"
    summary_df = pd.DataFrame(all_results_summary)

    # Sort by Quadratic Kappa (Standard for DAiSEE / Ordinal problems)
    summary_df = summary_df.sort_values(by="Kappa_Quad", ascending=False)

    summary_df.to_csv(summary_filename, index=False)
    print(f"\nAll training finished.")
    print(f"Master summary saved to: {summary_filename}")
    print(summary_df[['Model', 'Accuracy', 'MAE', 'Kappa_Quad']])
else:
    print("No models were trained successfully.")
