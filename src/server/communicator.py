import requests
from config import BROADCAST_INTERVAL, BASE_URL
from utils import compute_umap

class RestCommunicator:
    def __init__(self, url=BASE_URL, interval=BROADCAST_INTERVAL):
        self.url = url
        self.embeddings_url = f"{url}/latest_embeddings"
        self.embeddings_url_3d = f"{url}/3d_embeddings"
        self.annotation_url = f"{url}/annotate"
        self.metrics_url = f"{url}/metrics"

        self.interval = interval

    def maybe_send(self, iteration, embeddings, actual_labels, predicted_labels, filenames):
        if iteration % self.interval == 0:
            self.send_original(iteration, embeddings, actual_labels, predicted_labels, filenames)
            embeddings_3d = compute_umap(embeddings)
            self.send_3d(iteration, embeddings_3d, actual_labels, predicted_labels, filenames)

    def send_original(self, iteration, embeddings, actual_labels, predicted_labels, filenames):
        payload = {
            "iteration": iteration,
            "embedding_shape": list(embeddings.shape),
            "embeddings": embeddings.tolist(),
            "actual_labels": actual_labels,
            "predicted_labels": predicted_labels,
            "filenames": filenames
        }
        try:
            response = requests.post(self.embeddings_url, json=payload)
            response.raise_for_status()
            print(f"[RestCommunicator] Sent embeddings at iteration {iteration}")
        except requests.RequestException as e:
            print(f"[RestCommunicator] Error sending data: {e}")

    def send_3d(self, iteration, embeddings_3d, actual_labels, predicted_labels, filenames):
        payload = {
            "iteration": iteration,
            "embeddings": embeddings_3d.tolist(),
            "actual_labels": actual_labels,
            "predicted_labels": predicted_labels,
            "filenames": filenames
        }
        try:
            response = requests.post(self.embeddings_url_3d, json=payload)
            response.raise_for_status()
            print(f"[RestCommunicator] Sent 3D embeddings at iteration {iteration}")
        except requests.RequestException as e:
            print(f"[RestCommunicator] Error sending data: {e}")


    def send_annotation_request(self, filenames, indices):
        payload = {
            "filenames": filenames,
            "indices": indices
        }
        try:
            response = requests.post(self.annotation_url, json=payload)
            response.raise_for_status()
            print(f"[RestCommunicator] Sent {len(filenames)} samples for annotation")
        except requests.RequestException as e:
            print(f"[RestCommunicator] Error sending annotation request: {e}")

    def send_metrics(self, iteration, accuracy, loss):
        payload = {
            "iteration": iteration,
            "accuracy": accuracy,
            "loss": loss
        }
        try:
            response = requests.post(self.metrics_url, json=payload)
            response.raise_for_status()
            print(f"[RestCommunicator] Sent metrics at iteration {iteration}")
        except requests.RequestException as e:
            print(f"[RestCommunicator] Error sending metrics: {e}")

