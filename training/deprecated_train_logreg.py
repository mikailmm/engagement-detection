import datetime
import pathlib

import joblib
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

HOME_DIR = pathlib.Path.home()

mlflow.set_tracking_uri(f"sqlite:///{HOME_DIR}/.mlflow/mlflow.db")
mlflow.set_experiment("DAiSEE-Engagement")

DAISEE_ROOT = "~/Downloads/DAiSEE/"
DATA_ROOT = f"{DAISEE_ROOT}Data/"
LABEL_FOLDER = f"{DAISEE_ROOT}Labels/"


# Get current date and time for unique filenames (e.g., 2023-10-25_14-30-05)
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RESULTS_FILENAME = f"results/results_logreg_{current_time}.csv"
MODEL_FILENAME = f"models/model_logreg_{current_time}.pkl"

CSV_DATA_PATH = '../extract_engagement/facemesh_pose_2d_badan.csv'

df = pd.read_csv(CSV_DATA_PATH, header=0)

mlflow.data.pandas_dataset.from_pandas(
    df=df,
    source=CSV_DATA_PATH,
    targets="Engagement",
    name=CSV_DATA_PATH
)

# drop yang gak perlu
df = df.drop(columns=['person_id', 'Boredom', 'Confusion', 'Frustration '])

trainlabels = pd.read_csv(LABEL_FOLDER+'TrainLabels.csv', header=0)['ClipID'].astype(
    str).apply(lambda x: x.split(sep='.')[0]).unique()
testlabels = pd.read_csv(LABEL_FOLDER+'TestLabels.csv', header=0)['ClipID'].astype(
    str).apply(lambda x: x.split(sep='.')[0]).unique()
validationlabels = pd.read_csv(LABEL_FOLDER+'ValidationLabels.csv', header=0)[
    'ClipID'].astype(str).apply(lambda x: x.split(sep='.')[0]).unique()
# cleanlabels = testlabels['ClipID'].astype(
#     str).apply(lambda x: x.split(sep='.')[0]).unique()


# print(df['clip_id'].apply(lambda x: str(x) in testlabels))
# print(df['clip_id'].astype(str).isin(trainlabels).sum())
# print(df['clip_id'].astype(str).isin(testlabels).sum())
# print(df['clip_id'].astype(str).isin(validationlabels).sum())
# print(df['Engagement'].value_counts())

train = df[df['clip_id'].astype(str).isin(trainlabels)]
test = df[df['clip_id'].astype(str).isin(testlabels)]
validation = df[df['clip_id'].astype(str).isin(validationlabels)]
# print(train)
# print(test)
# print(validation)

# Here we finally drop 'clip_id' (metadata) and 'Engagement' (target) to get features
X_train = train.drop(columns=['filename', 'clip_id', 'Engagement'])
y_train = train['Engagement']

X_test = test.drop(columns=['filename', 'clip_id', 'Engagement'])
y_test = test['Engagement']

# Optional: Use validation set if needed, otherwise we test on Test set
X_val = validation.drop(columns=['filename', 'clip_id', 'Engagement'])
y_val = validation['Engagement']

mlflow.log_text(f"{train.columns}", 'train_columns.txt')

# # 6. TRAIN LOGISTIC REGRESSIONxxxxx
# # max_iter is increased because high-dimensional data (facemesh) often needs more steps to converge
# print("\nTraining Logistic Regression...")
# clf = LogisticRegression(random_state=42, max_iter=2000,
#                          solver='lbfgs')
# clf.fit(X_train, y_train)

# # 7. EVALUATION
# print("\n--- Evaluation on Test Set ---")
# y_pred = clf.predict(X_test)

# acc = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {acc:.4f}")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))


# Enable autologging for scikit-learn
mlflow.sklearn.autolog()

print(f"\nTraining Logistic Regression (Timestamp: {current_time})...")
# increased max_iter for convergence on high-dimensional data
clf = LogisticRegression(random_state=42, max_iter=1000,
                         solver='lbfgs', multi_class='auto')
clf.fit(X_train, y_train)
print("Training complete.")

# --- 7. EVALUATION ---
y_pred = clf.predict(X_test)

# Calculate Metrics
# We use average='weighted' because DAiSEE is a multi-class dataset (0,1,2,3) with class imbalance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
class_report_str = classification_report(y_test, y_pred)

mlflow.log_metrics(
    {'test_accuracy': accuracy,
     'test_precision': precision,
     'test_recall': recall,
     'test_f1': f1})
mlflow.log_text(class_report_str, 'classification_report.txt')

print("\n--- Results ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nDetailed Report:")
print(class_report_str)

# --- 8. SAVE RESULTS TO CSV ---
results_data = {
    'Date': [current_time],
    'Model': ['Logistic Regression'],
    'PathOrName': [CSV_DATA_PATH],
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1_Score': [f1],
    # Saves settings like C, max_iter, solver
    'Model_Params': [str(clf.get_params())],
    # Saves the full text report in one cell
    'Classification_Report': [class_report_str]
}

results_df = pd.DataFrame(results_data)
results_df.to_csv(RESULTS_FILENAME, index=False)
print(f"Performance metrics saved to {RESULTS_FILENAME}")

# --- 9. EXPORT MODEL ---
joblib.dump(clf, MODEL_FILENAME)
print(f"Model saved to {MODEL_FILENAME}")
