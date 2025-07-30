import os
from dataset.loader import UrbanSoundLoader
from dataset.labeled_manager import LabeledSetManager
from embeddings.pretrained_model import YAMNetEmbedder
from classifier.trainable_model import TrainableModel
from active_learning.active_learning_loop import UncertaintySampler, ActiveLearningLoop
from server.communicator import RestCommunicator
from utils import evaluate_model, load_annotations
import torch
from config import CHECKPOINT, BATCH_SIZE, NUM_ITERATIONS

if __name__ == '__main__':
    embedder = YAMNetEmbedder()
    loader = UrbanSoundLoader(embedder)
    model = TrainableModel(input_dim=1024, embedding_dim=64, num_classes=10)
    sampler = UncertaintySampler()
    communicator = RestCommunicator()

    fold_accuracies = []
    last_loss = 0
    labeled_indices, unlabeled_indices = [], []
    resume = False
    total_iterations = 0

    if os.path.exists(CHECKPOINT):
        checkpoint = torch.load(CHECKPOINT)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        total_iterations = checkpoint.get('total_iterations', 0)
        labeled_indices = checkpoint.get('labeled_indices', [])
        unlabeled_indices = checkpoint.get('unlabeled_indices', [])
        resume = True
        print(f"[main.py] Loaded checkpoint. Resuming from iteration {total_iterations}.")
        print(f"[main.py] Total labeled indices {len(labeled_indices)}.")
        print(f"[main.py] Total unabeled indices {len(unlabeled_indices)}.")
    else:
        print("[main.py] No checkpoint found. Starting from scratch.")

    # Loading human annotations if any
    human_annotations = load_annotations()

    for held_out_fold in range(1, 3):
        # Get data for current fold
        full_train_dataset, default_labeled, default_unlabeled, test_dataset, class_code_to_label = loader.get_labeled_unlabeled_datasets(
            held_out_fold)

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
        total_iterations, train_loss = loop.run(start_iteration=total_iterations, num_iters=NUM_ITERATIONS)
        #acc = evaluate_model(model, test_dataset)
        #print(f"[Fold {held_out_fold}] Accuracy: {acc:.4f}")
        #communicator.send_metrics(total_iterations, acc, last_loss)
        #fold_accuracies.append(acc)

    #avg_acc = sum(fold_accuracies)/len(fold_accuracies)
    #communicator.send_metrics(total_iterations, avg_acc, last_loss)
    #print(f"\nAverage Accuracy over all folds: {avg_acc:.4f}")
    # Save model state and optimizer
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
        'total_iterations': total_iterations,
    }, CHECKPOINT)




