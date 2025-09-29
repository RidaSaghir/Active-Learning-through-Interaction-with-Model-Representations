# dataset/cached_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset

class UrbanSoundCachedEmbeddingDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, labels, original_labels, filenames):
        # embeddings can be np.ndarray or np.memmap (mmap_mode='r')
        self.embeddings = embeddings
        self.labels = labels
        self.original_labels = original_labels
        self.filenames = filenames

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        emb = self.embeddings[idx].astype(np.float32)   # (1024,)
        return (
            torch.from_numpy(emb),
            torch.tensor(self.labels[idx], dtype=torch.long),
            self.original_labels[idx],
            self.filenames[idx],
            idx
        )
