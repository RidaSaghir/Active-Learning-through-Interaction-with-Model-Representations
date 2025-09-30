# simulate_baseline.py
import time, json, os
import requests
import pandas as pd
from config import DATA_DIR, HELD_OUT_FOLD, INITIAL_LABELS_PER_CLASS_COUNT, BASE_URL

API = BASE_URL.rstrip("/")
ANNOTATE_GET = f"{API}/annotate"
HUMAN_POST   = f"{API}/human_annotations"
CSV_PATH     = os.path.join(DATA_DIR, "UrbanSound8K.csv")

def build_index_to_code():
    """
    Reproduce the SAME index mapping your loader uses:
    1) make class codes on FULL df
    2) split train/test by held out fold
    3) sample PER_CLASS_COUNT per class for seed labeled
    4) concat labeled + unlabeled to form combined_df
    Return: dict[index_in_combined_df -> class_code]
    """
    df = pd.read_csv(CSV_PATH)
    # Step 1: build codes BEFORE splitting (matches your loader)
    df["class_code"] = df["class"].astype("category").cat.codes

    train_df = df[df["fold"] != HELD_OUT_FOLD]
    # labeled_df is the initial seed; unlabeled_df is the rest
    labeled_df = train_df.groupby("class_code", group_keys=False).apply(
        lambda x: x.sample(n=INITIAL_LABELS_PER_CLASS_COUNT, random_state=42)
    )
    unlabeled_df = train_df.drop(labeled_df.index)
    combined_df = pd.concat([labeled_df, unlabeled_df]).reset_index(drop=True)
    # map index -> class_code
    return combined_df["class_code"].to_dict(), combined_df["class"].to_list()

def get_annotation_items():
    """
    Support both response shapes:
      {"items":[{"index":123,"filename":"x.wav"}, ...]}
      ["x.wav", ...]   (legacy — not recommended)
    """
    r = requests.get(ANNOTATE_GET, timeout=3)
    r.raise_for_status()
    data = r.json()
    # New shape
    if isinstance(data, dict) and "items" in data:
        items = data["items"]
        idxs  = [int(d["index"]) for d in items]
        fns   = [d["filename"]    for d in items]
        return idxs, fns
    # Legacy shape
    if isinstance(data, list):
        # If server only returns filenames, we can't recover indices -> update server.
        raise RuntimeError("Server returned filenames only; please return indices too.")
    # Nothing to annotate
    return [], []

def post_human_annotations(indices, filenames, labels):
    payload = {
        "filenames": filenames,
        "indices":   indices,
        "labels":    labels,   # MUST be class codes (ints)
    }
    r = requests.post(HUMAN_POST, json=payload, timeout=3)
    r.raise_for_status()
    return r.json()

def main(poll_every=1.0):
    idx2code, _ = build_index_to_code()
    print("[baseline] ready; polling for annotation requests...")
    while True:
        try:
            idxs, fns = get_annotation_items()
            if idxs:
                lbls = [int(idx2code[int(i)]) for i in idxs]
                resp = post_human_annotations(idxs, fns, lbls)
                print(f"[baseline] labeled {len(idxs)} items | response={resp}")
            time.sleep(poll_every)
        except requests.RequestException as e:
            print(f"[baseline] API error: {e}")
            time.sleep(2.0)
        except Exception as e:
            print(f"[baseline] error: {e}")
            time.sleep(2.0)

if __name__ == "__main__":
    main()
