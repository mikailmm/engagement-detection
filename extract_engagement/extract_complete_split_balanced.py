import cv2
import mediapipe as mp
import csv
import os
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# --- CONFIGURATION ---
DISK = "/Volumes/Mika 128/DAiSEE/"
DATA_ROOT = f"{DISK}Data"
OUTPUT_CSV = "facemesh_pose_dataset_balanced.csv"

# Map the Split Name (Folder Name) to the specific Label File
SPLIT_CONFIG = {
    'Train': f"{DISK}Labels/TrainLabels.csv",
    'Test': f"{DISK}Labels/TestLabels.csv",
    'Validation': f"{DISK}Labels/ValidationLabels.csv"
}

# --- SAMPLING LEVELS ---
ENGAGEMENT_TO_FRAMES = {
    0: 300,
    1: 40,
    2: 5,
    3: 4
}

# Multiprocessing
NUM_PROCESSES = max(1, int(cpu_count() * 0.75))

# --- STEP 1: LOAD LABELS ---


def load_labels_dict(csv_path):
    """
    Loads labels for a specific split.
    """
    print(f"Loading labels from {os.path.basename(csv_path)}...")
    df = pd.read_csv(csv_path)

    # Clean header names (remove accidental spaces like 'Frustration ')
    df.columns = df.columns.str.strip()

    # Clean ClipID extensions
    df['clean_id'] = df['ClipID'].astype(
        str).apply(lambda x: os.path.splitext(x)[0])

    label_map = {}
    for _, row in df.iterrows():
        label_map[row['clean_id']] = {
            'engagement': int(row['Engagement']),
            # Ensure we access the clean column names
            'vector': [row['Boredom'], row['Engagement'], row['Confusion'], row['Frustration ']]
        }
    return label_map


# --- STEP 2: WORKER FUNCTION ---
mp_holistic = None


def init_worker():
    global mp_holistic
    mp_holistic = mp.solutions.holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        refine_face_landmarks=True
    )


def process_item(task):
    """
    task: (image_path, label_vector, person_id, clip_id, split_name)
    """
    img_path, labels, person_id, clip_id, split_name = task
    global mp_holistic

    try:
        image = cv2.imread(img_path)
        if image is None:
            return None

        results = mp_holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # --- BUILD ROW ---
        row = [os.path.basename(img_path)]
        row.append(person_id)
        row.append(clip_id)
        row.append(split_name)  # Optional: Useful to know if it was Train/Test
        row.extend(labels)

        # Face Mesh
        if results.face_landmarks:
            row.extend(
                [c for lm in results.face_landmarks.landmark for c in (lm.x, lm.y, lm.z)])
        else:
            row.extend([0.0] * 1434)

        # Pose
        if results.pose_landmarks:
            row.extend([c for lm in results.pose_landmarks.landmark for c in (
                lm.x, lm.y, lm.z, lm.visibility)])
        else:
            row.extend([0.0] * 132)

        return row

    except Exception:
        return None

# --- STEP 3: CRAWLER PER SPLIT ---


def get_tasks_for_split(split_name, label_file_path):
    """
    Crawls only the specific folder (e.g. Data/Train) using the specific label file.
    """
    tasks = []

    # 1. Load the specific labels for this split
    if not os.path.exists(label_file_path):
        print(f"Warning: Label file not found: {label_file_path}")
        return []

    label_map = load_labels_dict(label_file_path)

    # 2. Define the folder to scan
    split_folder = os.path.join(DATA_ROOT, split_name)

    print(f"Scanning {split_folder}...")

    for root, _, files in os.walk(split_folder):
        if not files:
            continue

        clip_id = os.path.basename(root)
        parent_dir = os.path.dirname(root)
        person_id = os.path.basename(parent_dir)

        # Match using the specific dictionary for this split
        if clip_id in label_map:
            data = label_map[clip_id]
            target_count = ENGAGEMENT_TO_FRAMES.get(data['engagement'], 5)

            # Sort files naturally
            raw_jpgs = [f for f in files if f.endswith('.jpg')]
            jpgs = sorted(raw_jpgs, key=lambda x: (len(x), x))

            total = len(jpgs)
            if total == 0:
                continue

            if total <= target_count:
                indices = range(total)
            else:
                indices = np.linspace(0, total - 1, target_count, dtype=int)

            for i in indices:
                full_path = os.path.join(root, jpgs[i])
                # Pass split_name to the worker
                tasks.append(
                    (full_path, data['vector'], person_id, clip_id, split_name))

    return tasks

# --- MAIN ---


def main():
    all_tasks = []

    # Iterate over Train, Test, Validation
    for split_name, label_path in SPLIT_CONFIG.items():
        split_tasks = get_tasks_for_split(split_name, label_path)
        print(f"Found {len(split_tasks)} frames in {split_name}")
        all_tasks.extend(split_tasks)

    print(f"Total frames to process: {len(all_tasks)}")

    if not all_tasks:
        print("No files found.")
        return

    # Headers
    headers = ['filename', 'person_id', 'clip_id', 'split',
               'Boredom', 'Engagement', 'Confusion', 'Frustration']
    headers.extend([f'f{i}_{ax}' for i in range(478) for ax in 'xyz'])
    headers.extend([f'p{i}_{ax}' for i in range(33) for ax in 'xyzv'])

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        with Pool(NUM_PROCESSES, initializer=init_worker) as pool:
            results = pool.imap_unordered(
                process_item, all_tasks, chunksize=25)

            for row in tqdm(results, total=len(all_tasks), unit="img"):
                if row:
                    writer.writerow(row)

    print("Done.")


if __name__ == '__main__':
    main()
