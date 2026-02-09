import numpy as np
import torch

from sklearn.metrics import accuracy_score, f1_score
from .logging_utils import get_logger

log = get_logger("imlvr.model_eval")

def evaluate_model_with_per_class(model, test_dataset, class_code_to_label=None, batch_size=64):
    """
    Works with BaseModel interface:
      - model.eval()
      - model.predict_proba(x) -> (B, C) numpy
    """
    log.info("Evaluating model on test dataset")
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

    num_classes = int(prob_np.shape[1]) if prob_np.size else 10
    cm = confusion_matrix_counts(y_true, y_pred, num_classes)

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

    return acc_global, avg_loss, per_class_acc, per_class_support, macro_f1, micro_f1, num_discovered, total_classes, discovered_classes, cm

def confusion_matrix_counts(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10):
    log.info("Computing confusion matrix")
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm
