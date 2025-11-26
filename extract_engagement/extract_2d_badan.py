import pandas as pd
import tqdm

# --- CONFIGURATION ---
INPUT_CSV = "facemesh_pose_dataset.csv"  # The 2.3GB file
OUTPUT_CSV = "facemesh_pose_2d_badan.csv"    # The new lighter file
# Rows to process at a time (RAM safe)
CHUNK_SIZE = 10000


def get_columns_to_keep(input_file):
    """
    Reads only the header row and determines which columns to keep
    based on your specific rules.
    """
    # 1. Read just the header (0 rows)
    all_cols = pd.read_csv(input_file, nrows=0).columns.tolist()

    keep_cols = []

    # Define exact pose columns to keep (Landmarks 11 and 12, X and Y only)
    # 11 = Left Shoulder, 12 = Right Shoulder
    target_pose_cols = ['p11_x', 'p11_y', 'p12_x', 'p12_y',
                        'p23_x', 'p23_y', 'p24_x', 'p24_y']

    for col in all_cols:
        # --- RULE 1: KEEP METADATA ---
        if col in ['filename', 'person_id', 'clip_id', 'Boredom', 'Engagement', 'Confusion', 'Frustration ']:
            keep_cols.append(col)
            continue

        # --- RULE 2: FACE MESH (Drop Z) ---
        if col.startswith('f'):
            # Only keep if it ends in _x or _y
            if col.endswith('_x') or col.endswith('_y'):
                keep_cols.append(col)
            continue

        # --- RULE 3: POSE (Drop Z, V, and non-11/12) ---
        if col.startswith('p'):
            if col in target_pose_cols:
                keep_cols.append(col)
            continue

    return keep_cols


def main():
    print("Analyzing columns...")
    cols_to_keep = get_columns_to_keep(INPUT_CSV)
    print(f"Original Column Count: Unknown (Likely ~1500+)")
    print(f"New Column Count: {len(cols_to_keep)}")
    print("Columns to keep (Preview):",
          cols_to_keep[:10], "...", cols_to_keep[-5:])

    print(f"\nProcessing {INPUT_CSV} in chunks...")

    # Open the output file in write mode first to clear it
    first_chunk = True

    # We use tqdm to show a progress bar.
    # Since we don't know total rows easily without reading whole file, we update by bytes or chunks.
    with pd.read_csv(INPUT_CSV, usecols=cols_to_keep, chunksize=CHUNK_SIZE) as reader:
        for chunk in tqdm.tqdm(reader, unit="chunk"):

            # Write mode: 'w' for first chunk (write header), 'a' for rest (append)
            mode = 'w' if first_chunk else 'a'
            header = first_chunk

            chunk.to_csv(OUTPUT_CSV, index=False, mode=mode, header=header)

            first_chunk = False

    print(f"Done! Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
