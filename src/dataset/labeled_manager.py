from torch.utils.data import DataLoader, Subset
import torch
import numpy as np

class LabeledSetManager:
    def __init__(self, full_dataset, labeled_indices, unlabeled_indices, batch_size=16):
        self.dataset = full_dataset
        self.labeled_indices = labeled_indices
        self.unlabeled_indices = unlabeled_indices
        self.batch_size = batch_size
        self._init_labeled_loader()

    def _init_labeled_loader(self):
        self.labeled_loader = DataLoader(
            Subset(self.dataset, self.labeled_indices),
            batch_size=self.batch_size,
            shuffle=True
        )
        self.labeled_iterator = iter(self.labeled_loader)

    def next_batch(self):
        try:
            batch = next(self.labeled_iterator)
        except StopIteration:
            # Restart the iterator with a fresh shuffle
            self._init_labeled_loader()
            batch = next(self.labeled_iterator)
        return batch

    def get_unlabeled_loader(self):
        """Return a DataLoader over current unlabeled data (used in uncertainty sampling)."""
        return DataLoader(Subset(self.dataset, self.unlabeled_indices), batch_size=self.batch_size)

    def add_from_unlabeled(self, indices):
        """Move given indices from unlabeled to labeled."""
        for idx in indices:
            if idx in self.unlabeled_indices:
                self.labeled_indices.append(idx)
                self.unlabeled_indices.remove(idx)

    def iter_labeled(self):
        """Yield (embedding, label, filename, index) for labeled data."""
        for i in self.labeled_indices:
            yield self.dataset[i]

    def iter_unlabeled(self):
        """Yield (embedding, dummy_label, filename, index) for unlabeled data."""
        for i in self.unlabeled_indices:
            yield self.dataset[i]
