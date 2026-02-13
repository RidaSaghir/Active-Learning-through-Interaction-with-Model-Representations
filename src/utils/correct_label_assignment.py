import json
import os
import pandas as pd
from pathlib import Path
from config import DATA_DIR
# ----------------------------
# CONFIG
# ----------------------------
ANNOTATION_PATH = "../moeez_annotations.json"
OUTPUT_JSON = "moeez_annotations_ideal.json"
metadata_path = os.path.join(DATA_DIR, "UrbanSound8K.csv")
# ----------------------------
# Load human annotations
# ----------------------------
with open(ANNOTATION_PATH, "r") as f:
    annotations = json.load(f)


# ----------------------------
# Load UrbanSound8K metadata
# ----------------------------
meta = pd.read_csv(metadata_path)

# Build fast lookup: filename -> classID
filename_to_label = dict(
    zip(meta["slice_file_name"], meta["classID"])
)

# ----------------------------
# Create oracle annotations
# ----------------------------
oracle_annotations = {}
missing_files = []

for sample_id, info in annotations.items():
    filename = info["filename"]

    if filename not in filename_to_label:
        missing_files.append(filename)
        continue

    true_label = int(filename_to_label[filename])

    oracle_annotations[sample_id] = {
        "label": true_label,
        "user": "oracle",
        "filename": filename
    }

# ----------------------------
# Save oracle file
# ----------------------------
with open(OUTPUT_JSON, "w") as f:
    json.dump(oracle_annotations, f, indent=2)

print(f"Saved oracle labels to: {OUTPUT_JSON}")
print(f"Total samples processed: {len(oracle_annotations)}")

if missing_files:
    print("\n⚠ Missing filenames in metadata:")
    for m in missing_files[:10]:
        print("  ", m)
    print(f"... total missing: {len(missing_files)}")