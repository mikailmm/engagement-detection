import cv2
import mediapipe as mp
import csv
import os
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# --- CONFIGURATION ---
DISK = "/Volumes/Mika 128/DAiSEE/"  # where the DAiSEE folder is saved
DATA_ROOT = f"{DISK}Data"
LABEL_FILE = f"{DISK}Labels/AllLabels.csv"
OUTPUT_CSV = "facemesh_pose_dataset.csv"

# --- SAMPLING LEVELS ---
ENGAGEMENT_TO_FRAMES = {
    0: 300,
    1: 40,
    2: 5,
    3: 4
}

# Multiprocessing
NUM_PROCESSES = max(1, int(cpu_count() * 0.75))

# --- STEP 1: LOAD LABELS & CLEAN EXTENSIONS ---


def load_labels_dict(csv_path):
    """
    Loads labels and strips extensions from ClipID to match folder names.
    """
    print("Loading labels...")
    df = pd.read_csv(csv_path)

    # CLEANING LOGIC:
    # The CSV has "video.avi" or "video.mp4". The folder is just "video".
    # We use os.path.splitext to drop ANY extension automatically.
    df['clean_id'] = df['ClipID'].astype(
        str).apply(lambda x: os.path.splitext(x)[0])

    label_map = {}
    for _, row in df.iterrows():
        # We map the Clean ID (Folder Name) to the data
        label_map[row['clean_id']] = {
            'engagement': int(row['Engagement']),
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
        refine_face_landmarks=True  # 478 landmarks
    )


def process_item(task):
    """
    task: (image_path, label_vector, person_id, clip_id)
    """
    # Unpack the new clip_id argument
    img_path, labels, person_id, clip_id = task
    global mp_holistic

    try:
        image = cv2.imread(img_path)
        if image is None:
            return None

        results = mp_holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # --- BUILD ROW ---
        # 1. Filename
        row = [os.path.basename(img_path)]

        # 2. Person ID
        row.append(person_id)

        # 3. Clip ID (NEW COLUMN)
        row.append(clip_id)

        # 4. Labels
        row.extend(labels)

        # 5. Face Mesh (478 points)
        if results.face_landmarks:
            row.extend(
                [c for lm in results.face_landmarks.landmark for c in (lm.x, lm.y, lm.z)])
        else:
            row.extend([0.0] * 1434)

        # 6. Pose (33 points)
        if results.pose_landmarks:
            row.extend([c for lm in results.pose_landmarks.landmark for c in (
                lm.x, lm.y, lm.z, lm.visibility)])
        else:
            row.extend([0.0] * 132)

        return row

    except Exception:
        return None

# --- STEP 3: CRAWLER ---


def get_dynamic_file_list(label_map):
    tasks = []
    print("Scanning directories (matching Folders to CSV)...")

    for root, _, files in os.walk(DATA_ROOT):
        if not files:
            continue

        clip_id = os.path.basename(root)
        parent_dir = os.path.dirname(root)
        person_id = os.path.basename(parent_dir)

        if clip_id in label_map:
            data = label_map[clip_id]
            target_count = ENGAGEMENT_TO_FRAMES.get(data['engagement'], 5)

            # --- CRITICAL FIX FOR SORTING ---
            # 1. Filter JPGs
            raw_jpgs = [f for f in files if f.endswith('.jpg')]

            # 2. Natural Sort: Sort by length first, then by character
            # This ensures frame '100' (len 3) comes after frame '99' (len 2)
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
                tasks.append((full_path, data['vector'], person_id, clip_id))

    return tasks


# --- MAIN ---
def main():
    # 1. Load Labels
    labels_map = load_labels_dict(LABEL_FILE)

    # 2. Get Files
    tasks = get_dynamic_file_list(labels_map)
    print(f"Selected {len(tasks)} frames.")

    if not tasks:
        print("No matching files found.")
        return

    # 3. Update Headers
    headers = ['filename', 'person_id', 'clip_id',
               'Boredom', 'Engagement', 'Confusion', 'Frustration ']
    headers.extend([f'f{i}_{ax}' for i in range(478) for ax in 'xyz'])
    headers.extend([f'p{i}_{ax}' for i in range(33) for ax in 'xyzv'])

    # 4. Run
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        with Pool(NUM_PROCESSES, initializer=init_worker) as pool:
            results = pool.imap_unordered(process_item, tasks, chunksize=25)

            for row in tqdm(results, total=len(tasks), unit="img"):
                if row:
                    writer.writerow(row)

    print("Done.")


if __name__ == '__main__':
    main()
