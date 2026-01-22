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

def evaluate_model_with_per_class(model, test_dataset, class_code_to_label=None, batch_size=64):
    """
    Works with BaseModel interface:
      - model.eval()
      - model.predict_proba(x) -> (B, C) numpy
    """
    model.eval()

    # simple torch loader
    loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    all_y = []
    all_pred = []
    all_prob = []

    total = 0
    correct = 0

    for x, y, *_ in loader:
        # predict_proba returns numpy
        prob = model.predict_proba(x)                  # [B, C] numpy
        pred = np.argmax(prob, axis=1)                 # [B]

        y_np = y.cpu().numpy()

        all_y.append(y_np)
        all_pred.append(pred)
        all_prob.append(prob)

        total += len(y_np)
        correct += (pred == y_np).sum()

    y_true = np.concatenate(all_y) if all_y else np.array([])
    y_pred = np.concatenate(all_pred) if all_pred else np.array([])
    prob_np = np.concatenate(all_prob) if all_prob else np.empty((0, 0))

    acc_global = float(correct / total) if total > 0 else 0.0

    # loss isn't meaningful for sklearn; keep it 0.0 for consistency
    avg_loss = 0.0

    # per-class accuracy
    per_class_acc = {}
    per_class_support = {}
    if y_true.size > 0:
        classes = np.unique(y_true)
        for c in classes:
            mask = (y_true == c)
            per_class_support[int(c)] = int(mask.sum())
            per_class_acc[int(c)] = float((y_pred[mask] == c).mean()) if mask.sum() else 0.0

    # F1 scores
    macro_f1 = float(f1_score(y_true, y_pred, average="macro")) if y_true.size else 0.0
    micro_f1 = float(f1_score(y_true, y_pred, average="micro")) if y_true.size else 0.0

    # discovered classes (based on predictions present)
    discovered_classes = sorted(list(set(map(int, y_pred.tolist())))) if y_pred.size else []
    num_discovered = len(discovered_classes)
    total_classes = int(prob_np.shape[1]) if prob_np.size else (len(class_code_to_label) if class_code_to_label else 0)

    # optional: map class ids -> labels for display
    if class_code_to_label:
        per_class_acc = {class_code_to_label[k]: v for k, v in per_class_acc.items()}
        per_class_support = {class_code_to_label[k]: v for k, v in per_class_support.items()}
        discovered_classes = [class_code_to_label[c] for c in discovered_classes]

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
