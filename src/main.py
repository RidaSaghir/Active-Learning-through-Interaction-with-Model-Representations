import os
import sys
import math
import threading
import torch
import uvicorn
import time
import random
import numpy as np

from dataset.loader import UrbanSoundLoader
from dataset.labeled_manager import LabeledSetManager
from embeddings.pretrained_model import YAMNetEmbedder
from classifier.model_factory import build_model
from active_learning.active_learning_loop import Sampler, ActiveLearningLoop
from server.communicator import RestCommunicator
from server.app import create_app
from server import state
from utils.misc import load_annotations
from utils.eval_funcs import evaluate_model_with_per_class
from utils.logging_utils import setup_logging, get_logger
from utils.curve import append_curve_row, save_curve_png
from config import (
    CHECKPOINT, BATCH_SIZE, HELD_OUT_FOLD,
    EPOCHS_BEFORE_QUERY, NUM_ANNOTATION_SUGGESTIONS,
    INITIAL_LABELS_PER_CLASS_COUNT, ACCURACY_TARGET,
    CSV_FILENAME, PNG_FILENAME,
    LABELS_PER_ROUND, MODEL_VARIANT, SEED,
    EPOCHS_FOR_RETRAING, OFFLINE_MODE
)

# -------------------------
# Determinism
# -------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def start_api_in_thread(host="0.0.0.0", port=8000):
    app = create_app()
    config = uvicorn.Config(app, host=host, port=port, log_config=None, access_log=False)
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    return t

def _read_annotations_for_mode(offline_cursor: int | None) -> dict:
    """
    Online: offline_cursor=None -> read whole file
    Offline: offline_cursor=int -> read prefix limited by cursor
    """
    if offline_cursor is None:
        return load_annotations()
    return load_annotations(limit=offline_cursor)

def run_trainer():
    log = get_logger("main")

    embedder = YAMNetEmbedder()
    loader = UrbanSoundLoader(embedder)

    model = build_model(
        variant=MODEL_VARIANT,
        input_dim=1024,
        num_classes=10
    )
    sampler = Sampler()
    communicator = RestCommunicator()
    log.info("Startup: created components | OFFLINE_MODE={OFFLINE_MODE} ")

    labeled_indices, unlabeled_indices = [], []
    resume = False
    total_iterations = 0

    # Offline cursor = how many labels are "visible" from the full JSON
    # Online mode: keep this as None and always use load_annotations() without limit.
    offline_cursor = 0 if OFFLINE_MODE else None

    if os.path.exists(CHECKPOINT):
        model.load(CHECKPOINT)
        meta = torch.load(CHECKPOINT + ".meta", map_location="cpu")
        total_iterations = meta.get("total_iterations", 0)
        labeled_indices = meta.get("labeled_indices", [])
        unlabeled_indices = meta.get("unlabeled_indices", [])
        if OFFLINE_MODE:
            offline_cursor = meta.get("offline_cursor", 0)
        resume = True
        log.info(
            f"Resumed from checkpoint | iter={total_iterations} "
            f"| labeled={len(labeled_indices)} | unlabeled={len(unlabeled_indices)} "
            f"| offline_cursor={offline_cursor}"
        )
    else:
        log.info("No checkpoint found; starting fresh")

    # --- Fixed held-out fold (constant test set) ---
    full_train_dataset, default_labeled, default_unlabeled, test_dataset, class_code_to_label = \
        loader.get_labeled_unlabeled_datasets(held_out_fold=HELD_OUT_FOLD)

    log.info(
        f"Split ready | train={len(full_train_dataset)} | "
        f"labeled_seed={len(default_labeled)} | unlabeled_seed={len(default_unlabeled)} | test={len(test_dataset)}"
    )
    state.class_code_to_label = class_code_to_label

    active_labeled = list(labeled_indices) if resume and labeled_indices else list(default_labeled)
    active_unlabeled = list(unlabeled_indices) if resume and unlabeled_indices else list(default_unlabeled)

    labeled_manager = LabeledSetManager(
        full_train_dataset,
        active_labeled,
        active_unlabeled,
        batch_size=BATCH_SIZE
    )

    loop = ActiveLearningLoop(labeled_manager, model, sampler, communicator, class_code_to_label)

    # Align seen annotations so old ones are not re-applied as "new"
    if resume:
        already = _read_annotations_for_mode(offline_cursor)
        loop._seen_annotations.update(already)

    # last_seen = number of labels currently "available" (online: file size; offline: visible prefix size)
    last_seen = len(_read_annotations_for_mode(offline_cursor))

    while True:

        added = loop._apply_new_annotations_and_train(
            epochs=EPOCHS_FOR_RETRAING,
            replay_fraction=1,
            annotations_limit=offline_cursor
        )
        if added:
            log.info(f"Applied new human annotations | +{added}")

        # -------------------------
        # 2) Train full passes (EPOCHS_BEFORE_QUERY epochs worth)
        # -------------------------
        steps_per_epoch = math.ceil(len(labeled_manager.labeled_indices) / BATCH_SIZE)
        num_iterations = steps_per_epoch * EPOCHS_BEFORE_QUERY

        total_iterations, last_loss = loop.run(
            start_iteration=total_iterations,
            num_iters=num_iterations
        )

        # -------------------------
        # 3) Evaluate + log
        # -------------------------
        (acc_global, avg_loss, per_class_acc, per_class_support, macro_f1, micro_f1,
         num_discovered, total_classes, discovered_classes, cm) = evaluate_model_with_per_class(
            model, test_dataset, class_code_to_label=class_code_to_label
        )

        log.info(
            f"Epoch end | iter={total_iterations} | "
            f"test_acc={acc_global:.4f} | test_loss={avg_loss:.4f}"
        )

        curve_csv = os.path.join("logs", f"{CSV_FILENAME}.csv")
        curve_png = os.path.join("logs", f"{PNG_FILENAME}.png")

        # Only count the labels that are currently "visible"
        human_labels_so_far = len(_read_annotations_for_mode(offline_cursor))
        total_labeled_now = len(labeled_manager.labeled_indices)
        labeled_counts = labeled_manager.get_labeled_counts_per_class(class_code_to_label)

        append_curve_row(
            curve_csv,
            iteration=total_iterations,
            total_labeled=total_labeled_now,
            human_labeled=human_labels_so_far,
            acc=acc_global,
            loss=avg_loss,
            macro_f1=macro_f1,
            micro_f1=micro_f1,
            num_discovered=num_discovered,
            total_classes=total_classes,
        )
        save_curve_png(
            curve_csv, curve_png,
            init_labels=INITIAL_LABELS_PER_CLASS_COUNT,
            human_annotations=NUM_ANNOTATION_SUGGESTIONS
        )

        communicator.send_metrics(
            iteration=total_iterations,
            accuracy=acc_global,
            loss=avg_loss,
            accuracy_target=ACCURACY_TARGET,
            total_labeled=total_labeled_now,
            human_labeled=human_labels_so_far,
            per_class_accuracy=per_class_acc,
            labeled_counts=labeled_counts,
            macro_f1=macro_f1,
            micro_f1=micro_f1,
            confusion_matrix=cm.tolist()
        )

        model.save(CHECKPOINT)
        meta_out = {
            "total_iterations": total_iterations,
            "labeled_indices": labeled_manager.labeled_indices,
            "unlabeled_indices": labeled_manager.unlabeled_indices,
        }
        if OFFLINE_MODE:
            meta_out["offline_cursor"] = offline_cursor
        torch.save(meta_out, CHECKPOINT + ".meta")
        log.info("Checkpoint saved")

        if acc_global >= ACCURACY_TARGET:
            sys.exit(0)

        while True:
            time.sleep(1.0)

            if OFFLINE_MODE:
                # Reveal next chunk from a pre-filled file
                offline_cursor = (offline_cursor or 0) + LABELS_PER_ROUND
                curr = _read_annotations_for_mode(offline_cursor)

                # File exhausted -> stop (offline replay finished)
                if len(curr) == last_seen:
                    log.info("No more labels to reveal (offline). Stopping.")
                    return

                # We only proceed when we crossed a full round boundary
                if len(curr) >= last_seen + LABELS_PER_ROUND:
                    last_seen = len(curr)
                    log.info(
                        f"Revealed offline labels | total={last_seen} "
                        f"| +{LABELS_PER_ROUND} -> retrain"
                    )
                    communicator.send_metrics(
                        iteration=total_iterations,
                        accuracy=0.0,
                        loss=0.0,
                        phase="retrain_start",
                        human_labeled=last_seen,
                    )
                    break

            else:
                # ONLINE: Wait until frontend appended LABELS_PER_ROUND new labels
                curr = _read_annotations_for_mode(None)
                if len(curr) >= last_seen + LABELS_PER_ROUND:
                    last_seen = len(curr)
                    log.info(
                        f"Detected new labels (online) | total={last_seen} "
                        f"| +{LABELS_PER_ROUND} -> retrain"
                    )
                    communicator.send_metrics(
                        iteration=total_iterations,
                        accuracy=0.0,
                        loss=0.0,
                        phase="retrain_start",
                        human_labeled=last_seen,
                    )
                    break


if __name__ == "__main__":
    setup_logging(filename="imlvr.log")
    log = get_logger("main")

    api_thread = start_api_in_thread(host="0.0.0.0", port=8000)
    log.info("FastAPI server started on http://0.0.0.0:8000")

    try:
        run_trainer()
        log.info("Training run finished")
    except KeyboardInterrupt:
        log.info("Interrupted by user")
