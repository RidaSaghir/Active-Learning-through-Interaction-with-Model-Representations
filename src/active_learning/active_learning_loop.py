import numpy as np
import torch
import torch.nn.functional as F
import itertools
from config import NUM_ANNOTATION_SUGGESTIONS
from utils.misc import load_annotations, diff_annotations
from utils.logging_utils import get_logger, log_duration



class UncertaintySampler:
    def select(self, unlabeled_loader, model, n=NUM_ANNOTATION_SUGGESTIONS):
        device = next(model.parameters()).device
        model.eval()
        all_entropies, all_indices = [], []
        with torch.no_grad():
            for x, _, _, _, idx in unlabeled_loader:
                x = x.to(device, non_blocking=True)
                logits, _ = model(x)
                logp = F.log_softmax(logits, dim=1)
                p = logp.exp()
                ent = -(p * logp).sum(dim=1)          # [B]
                all_entropies.append(ent.cpu())
                all_indices.append(idx)               # idx is already a tensor

        ent = torch.cat(all_entropies)                # [N_unlabeled]
        idx = torch.cat(all_indices)                  # [N_unlabeled]
        k = min(n, ent.numel())
        topk = torch.topk(ent, k=k, largest=True)
        return idx[topk.indices].tolist()



class ActiveLearningLoop:
    def __init__(self, manager, model, sampler, communicator, class_code_to_label):
        self.model = model
        self.sampler = sampler
        self.communicator = communicator
        self.manager = manager
        self.class_code_to_label = class_code_to_label
        self._seen_annotations = {}
        self.log = get_logger("imlvr.active_learning")


    def gather_full_dataset_view(self, manager):
        self.log.info("Gather full dataset view (for frontend)")
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

        # >>> make it an array here (and handle empty)
        emb_np = np.empty((0, self.model.classifier.in_features), np.float32) if len(embeddings) == 0 \
                else np.stack(embeddings, axis=0).astype(np.float32)

        self.log.info(f"Gathered view | total={len(filenames)} | dims={emb_np.shape[1]}")
        return emb_np, actual_labels, predicted_labels, filenames

    def _apply_new_annotations_and_train(self, device=None, epochs=1, replay_fraction=0.0):
        """Load human_annotations.json, take only NEW items, train on them."""
        current = load_annotations()
        delta = diff_annotations(current, self._seen_annotations)
        if not delta:
            return 0  # nothing to do

        # Update pools + labels
        self.manager.update_labels(delta)

        # Build a loader for just the new indices
        new_indices = list(delta.keys())
        new_loader = self.manager.get_loader_for_indices(new_indices, shuffle=True)

        # Optional replay to avoid forgetting: sample a few old labeled items
        if replay_fraction > 0 and len(self.manager.labeled_indices) > len(new_indices):
            import random
            old_pool = [i for i in self.manager.labeled_indices if i not in new_indices]
            k = max(1, int(len(new_indices) * replay_fraction))
            replay_idx = random.sample(old_pool, min(k, len(old_pool)))
            replay_loader = self.manager.get_loader_for_indices(replay_idx, shuffle=True)
        else:
            replay_loader = None

        # Train only on the delta (plus tiny replay if enabled)
        for _ in range(epochs):
            for x, y, _, _, _ in new_loader:
                loss, acc = self.model.train_step(x, y)
            if replay_loader:
                for x, y, _, _, _ in replay_loader:
                    _ = self.model.train_step(x, y)

        # Mark these as seen
        self._seen_annotations.update(delta)
        return len(delta)

    def run(self, start_iteration, num_iters):
        self.log.info(f"Run start | start_iter={start_iteration} | steps={num_iters}")
        global_iteration = start_iteration
        train_loss, train_accuracy = 0.0, 0.0

        for local_iteration in range(num_iters):
            x, y, _, filenames, idx = self.manager.next_batch()
            train_loss, train_accuracy = self.model.train_step(x, y)
            self.log.info(f"step={global_iteration} | batch={len(y)} | loss={train_loss:.4f} | acc={train_accuracy:.3f}")
            self.communicator.send_metrics(global_iteration, train_accuracy, train_loss)

            # throttle scene updates
            if local_iteration % 5 == 0:
                embeddings, actual, preds, fns = self.gather_full_dataset_view(self.manager)
                self.communicator.maybe_send(global_iteration, embeddings, actual, preds, fns)

            global_iteration += 1

        # Propose new items once, after the configured number of epochs
        unlabeled_loader = self.manager.get_unlabeled_loader()
        new_ids = self.sampler.select(unlabeled_loader, self.model)
        filenames_to_annotate = [self.manager.dataset[i][3] for i in new_ids]
        self.log.info(f"Suggest annotations | count={len(new_ids)}")
        self.communicator.send_annotation_request(filenames_to_annotate, new_ids)
        return global_iteration, train_loss

