# utils/curve.py
import os, csv, time
from config import NUM_CLASSES

def append_curve_row(csv_path, iteration, total_labeled, human_labeled, acc, loss):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "iteration", "total_labeled", "human_labeled", "accuracy", "loss"])
        w.writerow([int(time.time()), int(iteration), int(total_labeled), int(human_labeled),
                    float(acc), float(loss)])

# Optional: save a quick plot (no server deps)
def save_curve_png(csv_path, png_path, init_labels, human_annotations):
    import pandas as pd
    import matplotlib.pyplot as plt
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    if "human_labeled" not in df or "accuracy" not in df:
        return
    plt.figure()
    plt.plot(df["human_labeled"], df["accuracy"], marker="o")
    plt.xlabel("Human labels used")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.title(f"Accuracy vs. Labels (Initial labels = {init_labels*NUM_CLASSES}, Human Labels = {human_annotations})")
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
