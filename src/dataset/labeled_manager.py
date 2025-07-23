from torch.utils.data import DataLoader, Subset
import torch
import numpy as np
class LabeledSetManager:
    def __init__(self, labeled_dataset, unlabeled_dataset, batch_size=16):
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.batch_size = batch_size
        self.labeled_indices = list(range(len(labeled_dataset)))
        self.unlabeled_indices = list(range(len(unlabeled_dataset)))

    def next_batch(self):
        loader = DataLoader(Subset(self.labeled_dataset, self.labeled_indices), batch_size=self.batch_size, shuffle=True)
        return next(iter(loader))

    def get_unlabeled_loader(self):
        subset = Subset(self.unlabeled_dataset, self.unlabeled_indices)
        return DataLoader(subset, batch_size=self.batch_size)

    def add_from_unlabeled(self, indices):
        for idx in indices:
            if idx in self.unlabeled_indices:
                self.labeled_indices.append(idx)
                self.unlabeled_indices.remove(idx)

    def iter_labeled(self):
        for i in self.labeled_indices:
            yield self.labeled_dataset[i]

    def iter_unlabeled(self):
        for i in self.unlabeled_indices:
            yield self.unlabeled_dataset[i]

