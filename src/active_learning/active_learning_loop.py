import numpy as np
import torch
import torch.nn.functional as F
import itertools

class UncertaintySampler:
    def select(self, unlabeled_loader, model, n):
        uncertainties = []
        model.eval()
        with torch.no_grad():
            for x, idx in unlabeled_loader:
                logits, _ = model(x)
                probs = F.softmax(logits, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                for i, e in zip(idx, entropy):
                    uncertainties.append((i.item(), e.item()))
        uncertainties.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in uncertainties[:n]]



class ActiveLearningLoop:
    def __init__(self, trainer, sampler, communicator):
        self.trainer = trainer
        self.sampler = sampler
        self.communicator = communicator

    def gather_full_dataset_view(self, manager, model):
        embeddings, labels, filenames = [], [], []


        model.eval()
        with torch.no_grad():
            # Labeled
            for embedding_tensor, label_tensor, filename in manager.iter_labeled():
                _, z = model(embedding_tensor.unsqueeze(0))
                embeddings.append(z.squeeze(0).numpy())
                labels.append(label_tensor.item())
                filenames.append(filename)

            # Unlabeled
            for embedding_tensor, _, filename in manager.iter_unlabeled():
                logits, z = model(embedding_tensor.unsqueeze(0))
                pred_label = torch.argmax(logits, dim=1).item()
                embeddings.append(z.squeeze(0).numpy())
                labels.append(pred_label)
                filenames.append(filename)

        return np.stack(embeddings), np.array(labels), filenames

    def run(self, manager, model, num_iters=100):
        for i in range(num_iters):
            x, y, _ = manager.next_batch()
            loss = self.trainer.train_step(x, y)
            print(f"Iteration {i}, Loss: {loss:.4f}")

            embeddings, labels, filenames = self.gather_full_dataset_view(manager, model)
            self.communicator.maybe_send(i, embeddings, labels, filenames)

            if i % 10 == 0:
                unlabeled_loader = manager.get_unlabeled_loader()
                new_ids = self.sampler.select(unlabeled_loader, model, n=5)
                manager.add_from_unlabeled(new_ids)

