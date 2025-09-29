import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import BROADCAST_INTERVAL, BASE_URL
from utils.logging_utils import get_logger


class RestCommunicator:
    def __init__(self, url=BASE_URL, interval=BROADCAST_INTERVAL, timeout=3.0):
        self.log = get_logger("communicator")
        self.url = url.rstrip("/")
        self.embeddings_url = f"{url}/latest_embeddings"
        self.annotation_url = f"{url}/annotate"
        self.metrics_url = f"{url}/metrics"
        self.interval = interval
        self.timeout = timeout

        # Reuse TCP connections + basic retries
        self.session = requests.Session()
        retry = Retry(
            total=2, backoff_factor=0.2,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["POST", "GET"])
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def maybe_send(self, iteration, embeddings, actual_labels, predicted_labels, filenames):
        """Send embeddings every `interval` iterations."""
        if iteration % self.interval == 0:
            self.send_embeddings(iteration, embeddings, actual_labels, predicted_labels, filenames)

    def send_embeddings(self, iteration, embeddings, actual_labels, predicted_labels, filenames):
        payload = {
            "iteration": iteration,
            "embedding_shape": list(embeddings.shape),
            "embeddings": embeddings.tolist(),
            "actual_labels": actual_labels,
            "predicted_labels": predicted_labels,
            "filenames": filenames
        }
        try:
            r = self.session.post(self.embeddings_url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            self.log.info(f"Sent embeddings | iter={iteration} | shape={embeddings.shape}")
        except requests.RequestException as e:
            self.log.warning(f"Failed to send embeddings @iter={iteration}: {e}")

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
        payload = {"filenames": filenames, "indices": indices}
        try:
            r = self.session.post(self.annotation_url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            self.log.info(f"Sent annotation request | n={len(indices)}")
        except requests.RequestException as e:
            self.log.warning(f"Failed to send annotation request: {e}")

    def send_metrics(self, iteration, accuracy, loss):
        payload = {"iteration": iteration, "accuracy": accuracy, "loss": loss}
        try:
            r = self.session.post(self.metrics_url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            self.log.info(f"Sent metrics | iter={iteration} | acc={accuracy:.4f} | loss={loss:.4f}")
        except requests.RequestException as e:
            self.log.warning(f"Failed to send metrics @iter={iteration}: {e}")

