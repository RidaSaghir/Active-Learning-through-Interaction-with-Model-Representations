# Interactive Machine Learning with Active Learning and VR Interface

This project is a work in progress toward an interactive machine learning system for environmental sound classification using the UrbanSound8K dataset. It integrates active learning with pretrained audio embeddings (YAMNet) to enable efficient human-in-the-loop annotation and iterative model refinement. The system is designed to support interactive exploration, selective labeling, and incremental retraining within a unified experimental framework.

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

###  `/latest_embeddings`  
**GET** &  **POST**
Displays the latest raw embeddings in a json format.  
➡️ Visit this in your browser to view the embeddings, actual/predicted labels, and filenames.

---
###  `/3d_embeddings`  
**GET** &  **POST**
Displays the latest 3D embeddings in a json format.  (SUITABLE TO BE RENDERED IN FRONTEND)
➡️ Visit this in your browser to view the embeddings, actual/predicted labels, and filenames. 

---

###  `/table`  
**GET**  
Displays the latest embedding table in a browser-friendly HTML format.  
➡️ Visit this in your browser to view the embeddings, labels, and filenames in a tabular format.

---

### `/metrics`  
**GET** &  **POST**
Receives training metrics (accuracy and loss) from the backend training loop.  
You should see logs like  
`[FastAPI] Received metrics at iteration X` in your terminal.

---

### `/annotate`  
**GET** &  **POST**
Receives a list of filenames and indices for which user annotations are requested.

---

### `/plot`  
**GET**  
Renders an interactive plot for 3D/2D visualization of the intermediate embeddings.  
➡️ Visit this in your browser to explore.

---

### `/human_annotations`  
**POST**  
Accepts human-labeled indices and their corresponding labels.  (TO BE SENT VIA FRONTEND).
➡️ These are saved to `human_annotations.json` and used in future training cycles.

---

You can also explore these endpoints interactively at:  
[http://localhost:8000/docs](http://localhost:8000/docs)  
(FastAPI automatically generates Swagger UI)


---
## Payload Schemas
Please refer to src/server/schemas.py to see what is expected at different APIs.


