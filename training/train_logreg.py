import datetime
import pathlib

import joblib
import mlflow
import mlflow.data
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
)

# --- 1. CONFIGURATION & SETUP ---
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
RESULTS_FILENAME = RESULTS_DIR / f"results_logreg_{current_time}.csv"
MODEL_FILENAME = RESULTS_DIR / f"model_logreg_{current_time}.pkl"
REPORT_FILENAME = RESULTS_DIR / f"report_logreg_{current_time}.txt"

# --- 2. LOAD & PREPROCESS DATA ---
print("Loading data...")
df = pd.read_csv(CSV_DATA_PATH, header=0)

# Clean data
cols_to_drop = ['person_id', 'Boredom', 'Confusion', 'Frustration ']
df = df.drop(
    columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

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
# Keeping filename/clip_id in the dataframe for reference until X creation
feature_drop = ['filename', 'clip_id', 'Engagement']

X_train = train.drop(columns=feature_drop, errors='ignore')
y_train = train['Engagement']

X_test = test.drop(columns=feature_drop, errors='ignore')
y_test = test['Engagement']

X_val = validation.drop(columns=feature_drop, errors='ignore')
y_val = validation['Engagement']

print(y_train.value_counts())
print(y_test.value_counts())
print(y_val.value_counts())

print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

# --- 3. MLFLOW TRAINING RUN ---
print(f"\nStarting MLflow run (Timestamp: {current_time})...")

# Use context manager for safety
with mlflow.start_run(run_name=f"LogReg_{current_time}"):

    # A. Log Data Source
    # Create an MLflow dataset object from the original dataframe for lineage

    # abs_data_path = pathlib.Path(CSV_DATA_PATH).resolve()

    # dataset_source = mlflow.data.from_pandas(
    #     df,
    #     source=str(abs_data_path),
    #     targets="Engagement",
    #     name="DAiSEE_2D_Facemesh_Badan"
    # )
    # mlflow.log_input(dataset_source, context="train")

    # Log column names as a text artifact for reference
    mlflow.log_text(str(list(X_train.columns)), 'features.txt')

    # Set Tags
    mlflow.set_tag("model_type", "LogisticRegression")
    mlflow.set_tag("feature_set", "2D_Facemesh_Badan")
    mlflow.set_tag("algo_type", "Classification")

    # B. Enable Autologging
    # This automatically logs params, model, and TRAINING metrics
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        log_models=True
    )

    # C. Train Model
    print("Training Logistic Regression...")
    clf = LogisticRegression(
        random_state=42, max_iter=2000, solver='lbfgs')
    clf.fit(X_train, y_train)

    # D. Evaluate on TEST Set
    # Autolog generally calculates metrics on the training set (and val if provided).
    # We need to manually calculate and log TEST set metrics.
    print("Evaluating on Test Set...")
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    class_report_str = classification_report(y_test, y_pred)

    kappa = cohen_kappa_score(y_test, y_pred, weights='quadratic')

    # Log Test Metrics with a prefix to distinguish from training metrics
    mlflow.log_metrics({
        'test_accuracy_score': accuracy,
        'test_precision_score': precision,
        'test_recall_score': recall,
        'test_kappa_quadratic_score': kappa,
        'test_f1_score': f1})

    # Log the Classification Report as a text artifact
    mlflow.log_text(class_report_str, 'test_classification_report.txt')

    print("\n--- Test Results ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nDetailed Report:")
    print(class_report_str)

    # E. Save & Log Local Files
    # 1. Save Classification Report Text
    with open(REPORT_FILENAME, "w") as f:
        f.write(f"Date: {current_time}\n")
        f.write(class_report_str)
    mlflow.log_artifact(str(REPORT_FILENAME))  # Upload to MLflow

    # 2. Save Results CSV
    results_data = {
        'Date': [current_time],
        'Model': ['Logistic Regression'],
        'PathOrName': [CSV_DATA_PATH],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1_Score': [f1],
        'Model_Params': [str(clf.get_params())],
        'Classification_Report': [class_report_str]
    }
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(RESULTS_FILENAME, index=False)
    print(f"Metrics saved locally to {RESULTS_FILENAME}")
    mlflow.log_artifact(str(RESULTS_FILENAME))  # Upload to MLflow

    # 3. Save Model Locally (Autolog already saved it to MLflow, but this is for local backup)
    joblib.dump(clf, MODEL_FILENAME)
    print(f"Model saved locally to {MODEL_FILENAME}")

    print(
        f"Run Complete. View results in MLflow (Run ID: {mlflow.active_run().info.run_id})")
