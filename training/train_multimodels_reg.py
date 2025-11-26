import datetime
import pathlib

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR

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

# Global timestamp for this batch
BATCH_TIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# --- 2. LOAD & PREPROCESS DATA ---
print("Loading data...")
df = pd.read_csv(CSV_DATA_PATH, header=0)

# Clean data
cols_to_drop = ['person_id', 'Boredom', 'Confusion', 'Frustration ']
df = df.drop(
    columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')


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

# --- 3. DEFINE REGRESSION MODELS ---
models_config = [
    # (
    #     "Linear_Regression",
    #     LinearRegression()
    # ),
    # (
    #     "Elastic_Net",
    #     ElasticNet(random_state=42)
    # ),
    # (
    #     "Ridge_Regression",
    #     Ridge(alpha=1.0, random_state=42)
    # ),
    # (
    #     "Lasso_Regression",
    #     Lasso(alpha=0.1, random_state=42)
    # ),
    # (
    #     "RandomForest_Regressor",
    #     RandomForestRegressor(n_estimators=100, max_depth=15,
    #                           random_state=42, n_jobs=-1)
    # ),
    # (
    #     "GradientBoosting_Regressor",
    #     GradientBoostingRegressor(
    #         n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    # ),
    # # MLP (Neural Net) Regressor
    (
        "MLP_Regressor",
        MLPRegressor(hidden_layer_sizes=(100, ), max_iter=500, random_state=42,
                     verbose=True)
    )
]

# List to store summary
all_results_summary = []

# --- 4. TRAINING FUNCTION ---


def train_and_log_regressor(model_name, reg, X_train, y_train, X_test, y_test):
    print(f"\n[{model_name}] Starting training...")

    results_filename = RESULTS_DIR / f"results_{model_name}_{BATCH_TIME}.csv"
    model_filename = RESULTS_DIR / f"model_{model_name}_{BATCH_TIME}.pkl"
    report_filename = RESULTS_DIR / f"report_{model_name}_{BATCH_TIME}.txt"

    with mlflow.start_run(run_name=f"{model_name}_{BATCH_TIME}"):

        # Log basics
        mlflow.log_text(str(list(X_train.columns)), 'features.txt')
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("approach", "Regression_with_Rounding")
        mlflow.set_tag("algo_type", "Regression")

        # Enable Autologging (Works well for standard Regressors)
        mlflow.sklearn.autolog(
            log_input_examples=True,
            log_model_signatures=True,
            log_models=True
        )

        # Train
        reg.fit(X_train, y_train)
        print(f"[{model_name}] Training complete. Evaluating...")

        # --- PREDICTION & CONVERSION ---
        # 1. Raw continuous predictions (e.g., 2.34, 0.8, -0.1)
        y_pred_raw = reg.predict(X_test)

        # 2. Convert to Integers for Classification Metrics
        # Clip: Ensure values like -0.5 become 0, and 3.5 become 3
        # Round: 1.6 becomes 2, 1.4 becomes 1
        y_pred_clipped = np.clip(y_pred_raw, 0, 3)
        y_pred_class = np.round(y_pred_clipped).astype(int)

        # --- METRICS ---

        # A. Regression Metrics (How close is the continuous number?)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_raw))
        mae_raw = mean_absolute_error(y_test, y_pred_raw)

        # B. Classification Metrics (Based on rounded integers)
        accuracy = accuracy_score(y_test, y_pred_class)
        precision = precision_score(
            y_test, y_pred_class, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred_class, average='weighted')
        f1 = f1_score(y_test, y_pred_class, average='weighted')

        # C. Ordinal Metric
        kappa = cohen_kappa_score(y_test, y_pred_class, weights='quadratic')

        class_report_str = classification_report(y_test, y_pred_class)

        # Log Custom Metrics (Autolog handles basic R2/MSE, but we want Classification metrics too)
        mlflow.log_metrics({
            'test_rmse_score': rmse,
            'test_mae_score': mae_raw,
            'test_accuracy_score': accuracy,
            'test_precision_score': precision,
            'test_recall_score': recall,
            'test_f1_score': f1,
            'test_kappa_quadratic_score': kappa
        })

        mlflow.log_text(class_report_str, 'test_classification_report.txt')

        # --- GENERATE & LOG CONFUSION MATRIX ---
        print(f"[{model_name}] Generating Confusion Matrix...")

        # Calculate CM
        cm = confusion_matrix(y_test, y_pred_class, labels=[
                              0, 1, 2, 3], normalize='true')

        # Create Figure
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=[0, 1, 2, 3])
        disp.plot(cmap='Blues', ax=ax, values_format='.2f')

        ax.set_title(f"Confusion Matrix: {model_name}\n(Rounded Predictions)")

        # Log Figure to MLflow
        mlflow.log_figure(fig, "confusion_matrix.png")

        # Close plot to free memory
        plt.close(fig)

        print(
            f"[{model_name}] RMSE: {rmse:.4f} | Acc: {accuracy:.4f} | Kappa: {kappa:.4f}")

        # Save Artifacts Locally
        with open(report_filename, "w") as f:
            f.write(f"Date: {BATCH_TIME}\nModel: {model_name}\n")
            f.write(
                "Strategy: Regression -> Clip(0,3) -> Round -> Classification\n\n")
            f.write(class_report_str)
        mlflow.log_artifact(str(report_filename))

        # Save Model (Backup)
        joblib.dump(reg, model_filename)

        # CSV Results
        results_data = {
            'Date': [BATCH_TIME],
            'Model': [model_name],
            'PathOrName': [CSV_DATA_PATH],
            'RMSE_Raw': [rmse],
            'MAE_Raw': [mae_raw],
            'Accuracy_Rounded': [accuracy],
            'Kappa_Quad': [kappa],
            'F1_Score_Rounded': [f1],
            'Model_Params': [str(reg.get_params())],
            'Classification_Report': [class_report_str]
        }
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(results_filename, index=False)
        mlflow.log_artifact(str(results_filename))

        return results_data


# --- 5. MAIN EXECUTION LOOP ---
print(f"--- Starting Batch Regression: {len(models_config)} Models ---")

for model_name, model_instance in models_config:
    try:
        res = train_and_log_regressor(
            model_name, model_instance, X_train, y_train, X_test, y_test)

        summary_row = {k: v[0] for k, v in res.items()}
        all_results_summary.append(summary_row)

    except Exception as e:
        print(f"!!! Error training {model_name}: {e}")

# --- 6. SAVE MASTER SUMMARY ---
if all_results_summary:
    summary_filename = RESULTS_DIR / f"SUMMARY_REGRESSION_{BATCH_TIME}.csv"
    summary_df = pd.DataFrame(all_results_summary)

    # Sort by Quadratic Kappa (Higher is better) or RMSE (Lower is better)
    summary_df = summary_df.sort_values(by="Kappa_Quad", ascending=False)

    summary_df.to_csv(summary_filename, index=False)
    print(f"\nAll training finished.")
    print(f"Master summary saved to: {summary_filename}")
    # Display key metrics
    print(summary_df[['Model', 'RMSE_Raw', 'Accuracy_Rounded', 'Kappa_Quad']])
else:
    print("No models were trained successfully.")
