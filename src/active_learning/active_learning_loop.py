import numpy as np
import torch
import torch.nn.functional as F
import itertools
from config import NUM_ANNOTATION_SUGGESTIONS

class UncertaintySampler:
    def select(self, unlabeled_loader, model, n=NUM_ANNOTATION_SUGGESTIONS):
        uncertainties = []
        model.eval()
        with torch.no_grad():
            for x, _, _, idx in unlabeled_loader:
                logits, _ = model(x)
                probs = F.softmax(logits, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                for i, e in zip(idx, entropy):
                    uncertainties.append((i.item(), e.item()))
        uncertainties.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in uncertainties[:n]]



class ActiveLearningLoop:
    def __init__(self, manager, model, sampler, communicator):
        self.model = model
        self.sampler = sampler
        self.communicator = communicator
        self.manager = manager

    def gather_full_dataset_view(self, manager):
        embeddings, labels, filenames, label_types = [], [], [], []

        self.model.eval()
        with torch.no_grad():
            # Labeled
            for embedding_tensor, label_tensor, filename, index in manager.iter_labeled():
                logits, z = self.model(embedding_tensor.unsqueeze(0))
                pred_label = torch.argmax(logits, dim=1).item()
                embeddings.append(z.squeeze(0).numpy())
                labels.append(pred_label)
                filenames.append(filename)
                label_types.append("actual")

            # Unlabeled
            for embedding_tensor, _, filename, index in manager.iter_unlabeled():
                logits, z = self.model(embedding_tensor.unsqueeze(0))
                pred_label = torch.argmax(logits, dim=1).item()
                embeddings.append(z.squeeze(0).numpy())
                labels.append(pred_label)
                filenames.append(filename)
                label_types.append("predicted")

        return np.stack(embeddings), np.array(labels), filenames, label_types

    def run(self, start_iteration, num_iters=100):
        for local_iteration in range(num_iters):
            global_iteration = start_iteration + local_iteration
            x, y, filenames, idx = self.manager.next_batch()
            loss = self.model.train_step(x, y)
            print(f"Local Iteration {local_iteration}, Loss: {loss:.4f}")

            embeddings, labels, filenames, label_types = self.gather_full_dataset_view(self.manager)
            self.communicator.maybe_send(global_iteration, embeddings, labels, filenames, label_types)


        unlabeled_loader = self.manager.get_unlabeled_loader()
        new_ids = self.sampler.select(unlabeled_loader, self.model)
        # Gather metadata for these samples ([2] represents filename)
        filenames_to_annotate = [self.manager.dataset[i][2] for i in new_ids]
        self.communicator.send_annotation_request(filenames_to_annotate, new_ids)
        return global_iteration, loss

