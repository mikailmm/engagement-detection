import datetime
import pathlib

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.utils import resample

# --- 1. CONFIGURATION & SETUP ---
IS_RESAMPLED = False
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

# Global timestamp for this batch of experiments
BATCH_TIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# --- 2. LOAD & PREPROCESS DATA ---
print("Loading data...")
df = pd.read_csv(CSV_DATA_PATH, header=0)

# Clean data
cols_to_drop = ['person_id', 'Boredom', 'Confusion', 'Frustration ']
df = df.drop(
    columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

if IS_RESAMPLED:
    df = resample(df, n_samples=RESAMPLE_SIZE)


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

# X_val = validation.drop(columns=feature_drop, errors='ignore')
# y_val = validation['Engagement']

print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

# --- 3. DEFINE MODELS TO TRAIN ---
# Add or remove models here
models_config = [
    # (
    #     "LogisticRegression",
    #     LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
    # ),
    # (
    #     "RandomForest",
    #     RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    # ),
    # (
    #     "KNN",
    #     KNeighborsClassifier()
    # ),
    # (
    #     "DecisionTree",
    #     DecisionTreeClassifier(random_state=42)
    # ),
    # (
    #     "ExtraTrees",
    #     ExtraTreeClassifier(random_state=42)
    # )
    # (
    #     "GaussianNaïveBayes",
    #     GaussianNB()
    # )

    # HEAVY MODELS
    # (
    #     "GradientBoosting",
    #     GradientBoostingClassifier(
    #         n_estimators=100, learning_rate=0.1, random_state=42, verbose=1)
    # ),
    # # Uncomment SVM if dataset is not too huge (SVM is slow on large data)
    # (
    #     "SVM",
    #     SVC(kernel='rbf', probability=False, random_state=42, verbose=True)
    # ),

    # # Simple Neural Net
    (
        "MLP_NeuralNet",
        MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42,
                      verbose=True)
    ),
]

# List to store summary of all models for final comparison
all_results_summary = []

# --- 4. TRAINING FUNCTION ---


def train_and_log_model(model_name, clf, X_train, y_train, X_test, y_test):
    print(f"\n[{model_name}] Starting training...")

    # Unique filenames for this specific model
    results_filename = RESULTS_DIR / f"results_{model_name}_{BATCH_TIME}.csv"
    model_filename = RESULTS_DIR / f"model_{model_name}_{BATCH_TIME}.pkl"
    report_filename = RESULTS_DIR / f"report_{model_name}_{BATCH_TIME}.txt"

    # Start MLflow Run
    with mlflow.start_run(run_name=f"{'resampled_' if IS_RESAMPLED else ''}{model_name}_{BATCH_TIME}"):

        # Log column names
        mlflow.log_text(str(list(X_train.columns)), 'features.txt')

        # Tags
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("feature_set", "2D_Facemesh_Badan")
        mlflow.set_tag("algo_type", "Classification")

        # Enable Autologging
        mlflow.sklearn.autolog(
            log_input_examples=True,
            log_model_signatures=True,
            log_models=True
        )

        # Train
        clf.fit(X_train, y_train)
        print(f"[{model_name}] Training complete. Evaluating...")

        # Evaluate
        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        class_report_str = classification_report(y_test, y_pred)

        # Log Metrics
        mlflow.log_metrics({
            'test_accuracy_score': accuracy,
            'test_precision_score': precision,
            'test_recall_score': recall,
            'test_f1_score': f1
        })

        # Log Report Text
        mlflow.log_text(class_report_str, 'test_classification_report.txt')

        print(f"[{model_name}] Accuracy: {accuracy:.4f} | F1: {f1:.4f}")

        # Save Artifacts Locally & Upload
        # 1. Text Report
        with open(report_filename, "w") as f:
            f.write(f"Date: {BATCH_TIME}\nModel: {model_name}\n")
            f.write(class_report_str)
        mlflow.log_artifact(str(report_filename))

        # 2. Model (Manual Backup)
        joblib.dump(clf, model_filename)
        # Autolog handles model upload, but we can verify or upload custom if needed.
        # mlflow.log_artifact(str(model_filename))

        # 3. CSV Results (Individual)
        results_data = {
            'Date': [BATCH_TIME],
            'Model': [model_name],
            'PathOrName': [CSV_DATA_PATH],
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Recall': [recall],
            'F1_Score': [f1],
            'Model_Params': [str(clf.get_params())],
            'Classification_Report': [class_report_str]
        }
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(results_filename, index=False)
        mlflow.log_artifact(str(results_filename))

        # Return metrics for master summary
        return results_data


# --- 5. MAIN EXECUTION LOOP ---
print(f"--- Starting Batch Training: {len(models_config)} Models ---")
if IS_RESAMPLED:
    print(f"--- Using resampled data with {RESAMPLE_SIZE} samples")

for model_name, model_instance in models_config:
    try:
        # Run training function
        res = train_and_log_model(
            model_name, model_instance, X_train, y_train, X_test, y_test)

        # Flatten dictionary logic for summary appending
        # res is a dict of lists, we take the first element [0]
        summary_row = {k: v[0] for k, v in res.items()}
        all_results_summary.append(summary_row)

    except Exception as e:
        print(f"!!! Error training {model_name}: {e}")

# --- 6. SAVE MASTER SUMMARY ---
if all_results_summary:
    summary_filename = RESULTS_DIR / f"SUMMARY_ALL_{BATCH_TIME}.csv"
    summary_df = pd.DataFrame(all_results_summary)

    # Sort by F1 Score descending
    summary_df = summary_df.sort_values(by="F1_Score", ascending=False)

    summary_df.to_csv(summary_filename, index=False)
    print(f"\nAll training finished.")
    print(f"Master summary saved to: {summary_filename}")
    print(summary_df[['Model', 'Accuracy', 'F1_Score']])
else:
    print("No models were trained successfully.")
