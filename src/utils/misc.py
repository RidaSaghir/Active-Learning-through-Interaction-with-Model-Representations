import torch
import umap
import json
import os
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from config import HUMAN_ANNOTATIONS

def evaluate_model(model, dataset, batch_size=32):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size)
    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in dataloader:
            x, y, *rest = batch  # x: embeddings, y: labels
            logits, _ = model(x)
            preds = torch.argmax(logits, dim=1)
            loss = F.cross_entropy(logits, y, reduction='sum')
            total_loss += loss.item()
            total_count += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / total_count if total_count > 0 else 0.0
    return acc, avg_loss

def evaluate_model_with_per_class(model, dataset, batch_size=32, class_code_to_label=None):
    """
    Returns:
      acc_global: float
      per_class_acc: dict[class_label -> float]
      per_class_support: dict[class_label -> int]  (how many test samples)
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size)
    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in dataloader:
            x, y, *rest = batch  # x: embeddings, y: labels
            logits, _ = model(x)
            preds = torch.argmax(logits, dim=1)
            loss = F.cross_entropy(logits, y, reduction='sum')
            total_loss += loss.item()
            total_count += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc_global = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    micro_f1 = f1_score(all_labels, all_preds, average="micro")

    # Per-class using all labels
    per_class_acc = {}
    per_class_support = {}
    unique_classes = np.unique(all_labels)

    for c in unique_classes:
        idx = (all_labels == c)
        support = idx.sum()
        if support == 0:
            continue
        acc_c = (all_preds[idx] == all_labels[idx]).mean()
        label_name = class_code_to_label[int(c)] if class_code_to_label is not None else str(c)
        per_class_acc[label_name] = float(acc_c)
        per_class_support[label_name] = int(support)

    unique_preds = np.unique(all_preds)
    discovered_classes = [
        class_code_to_label[int(c)] if class_code_to_label else str(c)
        for c in unique_preds
    ]
    num_discovered = len(discovered_classes)
    total_classes = len(class_code_to_label) if class_code_to_label else len(unique_classes)

    avg_loss = total_loss / total_count if total_count > 0 else 0.0
    return acc_global, avg_loss, per_class_acc, per_class_support, macro_f1, micro_f1, num_discovered, total_classes, discovered_classes

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
