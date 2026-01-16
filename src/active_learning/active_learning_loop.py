import numpy as np
import torch
import torch.nn.functional as F
import itertools
from config import NUM_ANNOTATION_SUGGESTIONS, ANNOTATION_SUGGESTIONS_USING, PRODUCE_ANNOTATION_SUGGESTIONS
from utils.misc import load_annotations, diff_annotations
from utils.logging_utils import get_logger, log_duration
from active_learning.cue_computer import compute_cues_from_view



class UncertaintySampler:
    def select_from_cues(
        self,
        cues: dict,
        is_labeled_np: np.ndarray,
        indices: list,
        using: str,
        n: int = NUM_ANNOTATION_SUGGESTIONS,
    ):
        """
        Select top-n unlabeled indices according to a single cue.
        `cues` is the dict from compute_cues_from_view.
        `indices` are dataset indices aligned with cues arrays.
        """
        cue_key = using
        if cue_key not in cues:
            raise ValueError(f"Unknown cue '{using}'. Available: {list(cues.keys())}")

        scores_all = np.asarray(cues[cue_key], dtype=float)

        # Only consider unlabeled samples
        mask_unlabeled = ~is_labeled_np.astype(bool)      # [N]
        unlabeled_positions = np.where(mask_unlabeled)[0] # positions in arrays
        if len(unlabeled_positions) == 0:
            return []

        scores_u = scores_all[unlabeled_positions]
        k = min(n, len(scores_u))

        # higher score = more interesting → descending
        topk_rel = np.argsort(scores_u)[-k:]
        chosen_positions = unlabeled_positions[topk_rel]

        # Convert back to dataset indices (used by /annotate API)
        chosen_dataset_indices = [indices[i] for i in chosen_positions]
        return chosen_dataset_indices



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
        """
        Returns:
            emb_np:        [N, d] float32
            actual_labels: list length N (original labels if available)
            predicted:     list length N (string labels)
            filenames:     list length N
            indices:       list length N (dataset indices)
            prob_np:       [N, C] float32  model probabilities
            is_labeled:    np.bool_ [N]     True for labeled pool
            true_codes:    list length N
        """
        self.log.info("Gather full dataset view (for frontend)")

        embeddings = []
        probs = []
        predicted_labels = []
        actual_labels = []
        filenames = []
        indices = []
        is_labeled = []
        true_codes = []

        self.model.eval()

        # ---- LABELED SAMPLES ----
        with torch.no_grad():
            for embedding_tensor, label_tensor, original_label, filename, index in manager.iter_labeled():
                x = embedding_tensor.unsqueeze(0)  # [1, D]

                z = self.model.project(x)[0]  # numpy [d]
                p = self.model.predict_proba(x)[0]  # numpy [C]
                pred_code = int(p.argmax())

                embeddings.append(z)
                probs.append(p)
                predicted_labels.append(self.class_code_to_label[pred_code])
                actual_labels.append(original_label)
                filenames.append(filename)
                indices.append(int(index))
                is_labeled.append(True)
                true_codes.append(int(label_tensor.item()))

            # ---- UNLABELED SAMPLES ----
            for embedding_tensor, label_tensor, original_label, filename, index in manager.iter_unlabeled():
                x = embedding_tensor.unsqueeze(0)  # [1, D]

                z = self.model.project(x)[0]  # numpy [d]
                p = self.model.predict_proba(x)[0]  # numpy [C]
                pred_code = int(p.argmax())

                embeddings.append(z)
                probs.append(p)
                predicted_labels.append(self.class_code_to_label[pred_code])
                actual_labels.append(original_label)
                filenames.append(filename)
                indices.append(int(index))
                is_labeled.append(False)
                true_codes.append(int(label_tensor.item()))

        # ---- STACK OUTPUTS ----
        emb_np = (
            np.stack(embeddings).astype(np.float32)
            if embeddings else np.empty((0, 0), np.float32)
        )

        prob_np = (
            np.stack(probs).astype(np.float32)
            if probs else np.empty((0, 0), np.float32)
        )

        is_labeled_np = np.array(is_labeled, dtype=np.bool_)

        self.log.info(
            f"Gathered view | total={len(filenames)} | dims={emb_np.shape[1] if emb_np.size else 0}"
        )

        return (
            emb_np,
            actual_labels,
            predicted_labels,
            filenames,
            indices,
            prob_np,
            is_labeled_np,
            true_codes,
        )

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
            self.log.info(f"iteration={global_iteration} | batch={len(y)} | loss={train_loss:.4f} | acc={train_accuracy:.3f}")
            global_iteration += 1

        emb_np, actual_labels, predicted_labels, filenames, indices, prob_np, is_labeled, true_codes = self.gather_full_dataset_view(
            self.manager)
        cues = compute_cues_from_view(
            emb_np=emb_np,
            prob_np=prob_np,
            is_labeled_np=is_labeled,
            labeled_counts=self.manager.get_class_counts(),
            k_density=15,
        )
        self.communicator.send_embeddings(iteration=global_iteration, embeddings=emb_np, actual_labels=actual_labels,
                                     predicted_labels=predicted_labels, filenames=filenames, indices=indices,
                                     is_labeled=is_labeled, cues=cues, true_codes=true_codes)
        if PRODUCE_ANNOTATION_SUGGESTIONS:
            self.log.info(f"Computing suggestions based on {ANNOTATION_SUGGESTIONS_USING}")
            new_ids = self.sampler.select_from_cues(
                cues=cues,
                is_labeled_np=is_labeled,
                indices=indices,
                using=ANNOTATION_SUGGESTIONS_USING,
                n=NUM_ANNOTATION_SUGGESTIONS,
            )
            filenames_to_annotate = [self.manager.dataset[i][3] for i in new_ids]
            self.log.info(
                f"Suggest annotations | count={len(new_ids)} "
                f"| cue={ANNOTATION_SUGGESTIONS_USING}"
            )
            self.communicator.send_annotation_request(filenames_to_annotate, new_ids)

        return global_iteration, train_loss

