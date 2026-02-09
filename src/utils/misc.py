import umap
import json
import os
from config import HUMAN_ANNOTATIONS


def diff_annotations(current: dict, seen: dict):
    """Return only new or changed annotations."""
    delta = {}
    for idx, lbl in current.items():
        if idx not in seen or seen[idx] != lbl:
            delta[idx] = lbl
    return delta

def load_annotations(path=HUMAN_ANNOTATIONS):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    result = {}
    for k, v in data.items():
        if isinstance(v, dict):
            result[int(k)] = int(v.get("label"))
        else:
            result[int(k)] = int(v)
    return result

def compute_umap(embeddings, n_components: int = 3, random_state = 42):

    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    reduced = reducer.fit_transform(embeddings)
    return reduced
