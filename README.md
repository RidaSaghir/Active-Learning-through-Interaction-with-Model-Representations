# Interactive Machine Learning with Active Learning and VR Interface

This project is WORK IN PROGRESS for an interactive machine learning system for sound classification using the UrbanSound8K dataset. It leverages **active learning**, **pretrained audio embeddings (YAMNet)**, and a **VR-compatible backend** to support human-in-the-loop annotation and retraining.

---
## How to get it working?
### 1. Clone the Repository

Use one of the following commands:

```bash
git clone https://git.ni.dfki.de/iml/cst/long-time-scale/interactive-machine-learning.git
# OR
git clone git@git.ni.dfki.de:iml/cst/long-time-scale/interactive-machine-learning.git
```

### 2. Download the dataset
- We are currently using [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) for this project which can be downloaded from https://urbansounddataset.weebly.com/download-urbansound8k.html.
- You’ll be redirected to fill out a short form before the download begins.

### 3. Build and run the docker image

- Open a terminal in the root directory of the project (where Dockerfile is located) and run:
```bash
docker build -t interactive-ml-app .      
# Replace <ABSOLUTE_PATH_TO_DATASET> with your actual dataset path
docker run \
  -v <ABSOLUTE_PATH_TO_DATASET>:/data \
  -e DATA_DIR=/data \
  -p 8000:8000 \
  interactive-ml-app
```
---
## API Routes

Once the Docker container is running, your FastAPI backend will expose the following HTTP routes at [http://localhost:8000](http://localhost:8000):

---

###  `/table`  
**GET**  
Displays the latest embedding table in a browser-friendly HTML format.  
➡️ Visit this in your browser to view the embeddings, labels, and filenames in a tabular format.

---

### `/metrics`  
**POST**  
Receives training metrics (accuracy and loss) from the backend training loop.  
You should see logs like  
`[FastAPI] Received metrics at iteration X` in your terminal.

---

### 🧠 `/annotate`  
**POST**  
Receives a list of filenames and indices for which user annotations are requested.  

### `/annotate`  
**GET**  
Fetches the current list of files that require human annotation.  
➡️ Visit this in your browser or via frontend to see which files the system wants labeled.

---

### `/human_annotations`  
**POST**  
Accepts human-labeled indices and their corresponding labels.  
➡️ These are saved to `human_annotations.json` and used in future training cycles.

---

You can also explore these endpoints interactively at:  
[http://localhost:8000/docs](http://localhost:8000/docs)  
(FastAPI automatically generates Swagger UI)


## Project Structure & Class Responsibilities

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
  - Projects them into a **trainable embedding space** (64D).
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



