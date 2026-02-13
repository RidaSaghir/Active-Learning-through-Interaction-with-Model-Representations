import pandas as pd
import os
import json
from config import DATA_DIR

with open("../moeez_annotations.json") as f:
    human_annotations = json.load(f)

metadata_path = os.path.join(DATA_DIR, "UrbanSound8K.csv")
df = pd.read_csv(metadata_path)

filename_to_true_code = {
    row["slice_file_name"]: int(row["classID"])
    for _, row in df.iterrows()
}

correct = 0
total = 0

for entry in human_annotations.values():
    fname = entry["filename"]
    human_label = entry["label"]
    true_label = filename_to_true_code.get(fname)

    if true_label is not None:
        if human_label == true_label:
            correct += 1
        total += 1

human_accuracy = correct / total if total > 0 else None
print(f"Human labeling accuracy: {human_accuracy:.2%}")
