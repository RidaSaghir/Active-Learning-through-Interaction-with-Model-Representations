import torch
import umap
import json
import os
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from config import HUMAN_ANNOTATIONS
def evaluate_model(model, dataset, batch_size=32):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            x, y, _, _ = batch  # x: embeddings, y: labels
            logits, _ = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return acc

def load_annotations(path=HUMAN_ANNOTATIONS):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}

def compute_umap(embeddings, n_components: int = 3, random_state = 42):

    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    reduced = reducer.fit_transform(embeddings)
    return reduced
