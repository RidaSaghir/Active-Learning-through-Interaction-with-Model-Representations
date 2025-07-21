from dataset.loader import UrbanSoundLoader
from dataset.labeled_manager import LabeledSetManager
from embeddings.pretrained_model import YAMNetEmbedder
from classifier.trainable_model import TrainableModel
from active_learning.active_learning_loop import UncertaintySampler, ActiveLearningLoop
from server.communicator import RestCommunicator
from config import EMBEDDING_DIM, NUM_CLASSES

if __name__ == '__main__':
    embedder = YAMNetEmbedder()
    loader = UrbanSoundLoader(embedder)

    # Now returns datasets, not loaders
    labeled_ds, unlabeled_ds = loader.get_labeled_unlabeled_datasets()

    # Use manager to dynamically update and serve batches
    labeled_manager = LabeledSetManager(labeled_ds, unlabeled_ds, batch_size=16)

    model = TrainableModel(input_dim=1024, embedding_dim=64, num_classes=10)
    sampler = UncertaintySampler()
    communicator = RestCommunicator()

    loop = ActiveLearningLoop(model, sampler, communicator)
    loop.run(labeled_manager, model)
