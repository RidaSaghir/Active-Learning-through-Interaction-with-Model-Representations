import os
import sys
import math
import threading
import torch
import uvicorn
import time

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
from config import (CHECKPOINT, BATCH_SIZE, HELD_OUT_FOLD, EPOCHS_BEFORE_QUERY, NUM_ANNOTATION_SUGGESTIONS,
                    INITIAL_LABELS_PER_CLASS_COUNT, ACCURACY_TARGET, CSV_FILENAME, PNG_FILENAME,
                    LABELS_PER_ROUND, MODEL_VARIANT)

def start_api_in_thread(host="0.0.0.0", port=8000):
    app = create_app()
    # Use the same logging you set up below; disable uvicorn’s own config
    config = uvicorn.Config(app, host=host, port=port, log_config=None, access_log=False)
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    return t

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
    log.info("Startup: created components")

    last_loss = 0.0
    labeled_indices, unlabeled_indices = [], []
    resume = False
    total_iterations = 0

    if os.path.exists(CHECKPOINT):
        model.load(CHECKPOINT)
        meta = torch.load(CHECKPOINT + ".meta")
        total_iterations = meta.get("total_iterations", 0)
        labeled_indices = meta.get("labeled_indices", [])
        unlabeled_indices = meta.get("unlabeled_indices", [])
        resume = True
        log.info(
            f"Resumed from checkpoint | iter={total_iterations} "
            f"| labeled={len(labeled_indices)} | unlabeled={len(unlabeled_indices)}"
        )
    else:
        log.info("No checkpoint found; starting fresh")

    # Loading human annotations if any
    human_annotations = load_annotations()
    log.info(f"Loaded human annotations: {len(human_annotations)}")

    # --- Fixed held-out fold (constant test set) ---
    full_train_dataset, default_labeled, default_unlabeled, test_dataset, class_code_to_label = \
        loader.get_labeled_unlabeled_datasets(held_out_fold=HELD_OUT_FOLD)
    log.info(f"Split ready | train={len(full_train_dataset)} | labeled_seed={len(default_labeled)} | unlabeled_seed={len(default_unlabeled)} | test={len(test_dataset)}")
    state.class_code_to_label = class_code_to_label


    # Choose pools based on checkpoint or defaults
    active_labeled = labeled_indices if resume and labeled_indices else default_labeled
    active_unlabeled = unlabeled_indices if resume and unlabeled_indices else default_unlabeled

    # Apply human annotations to dataset if available
    if human_annotations:
        for idx, label in human_annotations.items():
            full_train_dataset.labels[idx] = label
            if idx not in active_labeled:
                active_labeled.append(idx)
            if idx in active_unlabeled:
                active_unlabeled.remove(idx)

    labeled_manager = LabeledSetManager(
            full_train_dataset,
            active_labeled,
            active_unlabeled,
            batch_size=BATCH_SIZE
        )

    loop = ActiveLearningLoop(labeled_manager, model, sampler, communicator, class_code_to_label)
    last_seen = len(load_annotations())
    while True:
        # Incorporate any NEW labels once, before training this cycle
        added = loop._apply_new_annotations_and_train(epochs=1, replay_fraction=1)
        if added:
            log.info(f"Applied new human annotations | +{added}")

        # 2) Train for X full epochs, then query once
        steps_per_epoch = math.ceil(len(labeled_manager.labeled_indices) / BATCH_SIZE)
        num_iterations = steps_per_epoch * EPOCHS_BEFORE_QUERY
        total_iterations, last_loss = loop.run(
            start_iteration=total_iterations,
            num_iters=num_iterations  # <-- drives X full passes
        )

        # 3) Evaluate + send + checkpoint
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

        # how many human labels are in play right now
        human_labels_so_far = len(load_annotations())
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
        save_curve_png(curve_csv, curve_png, init_labels=INITIAL_LABELS_PER_CLASS_COUNT, human_annotations=NUM_ANNOTATION_SUGGESTIONS)
        log.info(f"Sent metrics to {curve_csv}")
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

        torch.save({
            "total_iterations": total_iterations,
            "labeled_indices": labeled_manager.labeled_indices,
            "unlabeled_indices": labeled_manager.unlabeled_indices,
        }, CHECKPOINT + ".meta")

        log.info("Checkpoint saved")

        log.info("Checkpoint saved")

        if acc_global >= ACCURACY_TARGET: sys.exit(0)

        # 4) Wait for more labels
        while True:
            time.sleep(1.0)
            curr = load_annotations()
            if len(curr) >= last_seen + LABELS_PER_ROUND:  # new labels arrived
                last_seen = len(curr)
                log.info(
                    f"Detected new labels | total={last_seen} "
                    f"| +{LABELS_PER_ROUND} since last round -> retrain"
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

    # Start API in background
    api_thread = start_api_in_thread(host="0.0.0.0", port=8000)
    log.info("FastAPI server started on http://0.0.0.0:8000")

    # Run trainer (blocking)
    try:
        run_trainer()
        log.info("Training run finished")
    except KeyboardInterrupt:
        log.info("Interrupted by user")





