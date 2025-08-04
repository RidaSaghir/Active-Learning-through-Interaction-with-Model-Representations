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
            for x, _, _,_, idx in unlabeled_loader:
                logits, _ = model(x)
                probs = F.softmax(logits, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                for i, e in zip(idx, entropy):
                    uncertainties.append((i.item(), e.item()))
        uncertainties.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in uncertainties[:n]]



class ActiveLearningLoop:
    def __init__(self, manager, model, sampler, communicator, class_code_to_label):
        self.model = model
        self.sampler = sampler
        self.communicator = communicator
        self.manager = manager
        self.class_code_to_label = class_code_to_label

    def gather_full_dataset_view(self, manager):
        embeddings, predicted_labels, actual_labels, filenames, label_types = [], [], [], [], []

        self.model.eval()
        with torch.no_grad():
            # Labeled
            for embedding_tensor, label_tensor, original_label, filename, index in manager.iter_labeled():
                logits, z = self.model(embedding_tensor.unsqueeze(0))
                pred_code = torch.argmax(logits, dim=1).item()
                pred_label = self.class_code_to_label[pred_code]
                embeddings.append(z.squeeze(0).numpy())
                predicted_labels.append(pred_label)
                actual_labels.append(original_label)
                filenames.append(filename)

            # Unlabeled
            for embedding_tensor, label_tensor, original_label, filename, index in manager.iter_unlabeled():
                logits, z = self.model(embedding_tensor.unsqueeze(0))
                pred_code = torch.argmax(logits, dim=1).item()
                pred_label = self.class_code_to_label[pred_code]
                embeddings.append(z.squeeze(0).numpy())
                predicted_labels.append(pred_label)
                actual_labels.append(original_label)
                filenames.append(filename)

        return np.stack(embeddings), actual_labels, predicted_labels, filenames

    def run(self, start_iteration, num_iters=100):
        for local_iteration in range(num_iters):
            global_iteration = start_iteration + local_iteration
            x, y, _, filenames, idx = self.manager.next_batch()
            train_loss, train_accuracy = self.model.train_step(x, y)
            self.communicator.send_metrics(global_iteration, train_accuracy, train_loss)
            print(f"Local Iteration {local_iteration}, Loss: {train_loss:.4f}")
            embeddings, actual_labels, predicted_labels, filenames = self.gather_full_dataset_view(self.manager)
            self.communicator.maybe_send(global_iteration, embeddings,  actual_labels, predicted_labels, filenames)

        unlabeled_loader = self.manager.get_unlabeled_loader()
        new_ids = self.sampler.select(unlabeled_loader, self.model)
        # Gather metadata for these samples ([2] represents filename)
        filenames_to_annotate = [self.manager.dataset[i][2] for i in new_ids]
        self.communicator.send_annotation_request(filenames_to_annotate, new_ids)
        return global_iteration, train_loss

