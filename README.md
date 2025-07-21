# Interactive Machine Learning with Active Learning and VR Interface

This project is WORK IN PROGRESS for an interactive machine learning system for sound classification using the UrbanSound8K dataset. It leverages **active learning**, **pretrained audio embeddings (YAMNet)**, and a **VR-compatible backend** to support human-in-the-loop annotation and retraining.

---

## 📁 Project Structure & Class Responsibilities

### `embeddings/pretrained_model.py`

#### `YAMNetEmbedder`
- Loads the [YAMNet](https://tfhub.dev/google/yamnet/1) model from TensorFlow Hub.
- Extracts 1024-dimensional audio embeddings from raw audio waveforms.
- Embeddings are **frozen** and used as input features to a trainable classifier.

---

### `classifier/trainable_model.py`

#### `TrainableModel`
- A PyTorch model that:
  - Takes frozen YAMNet embeddings as input.
  - Projects them into a **trainable embedding space** (256D).
  - Applies a classification head for supervised learning.
- Returns both:
  - `logits` (for classification)
  - `z` (learned trainable embedding used for VR visualization).

---

### `active_learning/active_learning_loop.py`

#### `UncertaintySampler`
- Selects the most uncertain samples (based on entropy of predictions).
- Used for active learning — adds informative unlabeled samples into the labeled set.

#### `ActiveLearningLoop`
- Orchestrates the full loop:
  1. Train on currently labeled data.
  2. Predict on unlabeled data.
  3. Select high-uncertainty samples and promote to labeled set.
  4. Broadcast improved embeddings and predictions to VR frontend.

---

### `dataset/loader.py`

#### `UrbanSoundLoader`
- Loads metadata and paths from UrbanSound8K.
- Splits into labeled and unlabeled samples.
- Returns PyTorch datasets that include:
  - YAMNet embedding
  - True label (or placeholder)
  - Filename (for tracking and user display)

---

### `dataset/labeled_manager.py`

#### `LabeledSetManager`
- Manages labeled and unlabeled indices dynamically.
- Provides:
  - `next_batch()` – randomly sampled batch from labeled data
  - `add_from_unlabeled()` – add new samples into the labeled pool
  - `get_unlabeled_loader()` – return a loader for uncertainty sampling
  - `get_labeled_embeddings_labels_filenames()` – used to collect current training state for VR

---

### `server/communicator.py`

#### `RestCommunicator`
- Sends embeddings + predicted labels + filenames to FastAPI backend every few iterations (`BROADCAST_INTERVAL`).
- Acts as a bridge between training loop and VR frontend.

---

### `server/app.py`

- **FastAPI backend** that receives and stores the latest embeddings.
- Exposes the following endpoints:
  - `POST /embeddings` – accepts new data from the training loop.
  - `GET /latest_embeddings` – returns raw embeddings + predicted labels.
  - `GET /plot` – returns a PCA 2D plot.
  - `GET /table` – returns an HTML table view of embeddings + labels + filenames.

---



