from torch.utils.data import DataLoader, Subset
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

    def get_embeddings_and_labels(self):
        embeddings, labels, filenames = [], [], []
        for i in self.labeled_indices:
            embedding_tensor, label_tensor, filename = self.labeled_dataset[i]
            embeddings.append(embedding_tensor.numpy())
            labels.append(label_tensor.item())
            filenames.append(filename)
        return embeddings, labels, filenames