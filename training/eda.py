import pandas as pd
import sklearn

DAISEE_ROOT = "~/Downloads/DAiSEE/"
DATA_ROOT = f"{DAISEE_ROOT}Data/"
LABEL_FOLDER = f"{DAISEE_ROOT}Labels/"


df = pd.read_csv(
    '../extract_engagement/facemesh_pose_cleaned_2d.csv', header=0)
trainlabels = pd.read_csv(LABEL_FOLDER+'TrainLabels.csv', header=0)['ClipID'].astype(
    str).apply(lambda x: x.split(sep='.')[0]).unique()
testlabels = pd.read_csv(LABEL_FOLDER+'TestLabels.csv', header=0)['ClipID'].astype(
    str).apply(lambda x: x.split(sep='.')[0]).unique()
validationlabels = pd.read_csv(LABEL_FOLDER+'ValidationLabels.csv', header=0)[
    'ClipID'].astype(str).apply(lambda x: x.split(sep='.')[0]).unique()
# cleanlabels = testlabels['ClipID'].astype(
#     str).apply(lambda x: x.split(sep='.')[0]).unique()
print(df)
print(trainlabels)
print(testlabels)
print(validationlabels)
# print(df['clip_id'].apply(lambda x: str(x) in testlabels))
print(df['clip_id'].astype(str).isin(trainlabels).sum())
print(df['clip_id'].astype(str).isin(testlabels).sum())
print(df['clip_id'].astype(str).isin(validationlabels).sum())
print(df['Engagement'].value_counts())
