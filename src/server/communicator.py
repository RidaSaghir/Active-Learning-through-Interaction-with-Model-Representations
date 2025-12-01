import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import BROADCAST_INTERVAL, BASE_URL
from utils.logging_utils import get_logger


class RestCommunicator:
    def __init__(self, url=BASE_URL, interval=BROADCAST_INTERVAL, timeout=3.0):
        self.log = get_logger("communicator")
        self.url = url.rstrip("/")
        self.embeddings_url = f"{self.url}/latest_embeddings"
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

    def maybe_send(self, iteration, embeddings, actual_labels, predicted_labels, filenames, indices, is_labeled, cues, true_codes):
        """Send embeddings every `interval` iterations."""
        if iteration % self.interval == 0:
            self.send_embeddings(iteration, embeddings, actual_labels, predicted_labels, filenames, indices, is_labeled, cues, true_codes)

    def send_embeddings(self, iteration, embeddings, actual_labels, predicted_labels, filenames, indices, is_labeled, cues, true_codes):
        payload = {
            "iteration": iteration,
            "embedding_shape": list(embeddings.shape),
            "embeddings": embeddings.tolist(),
            "actual_labels": actual_labels,
            "predicted_labels": predicted_labels,
            "filenames": filenames,
            "indices": indices,
            "is_labeled": is_labeled.tolist(),
            "cues": {name: arr.tolist() for name, arr in cues.items()},
            "true_codes": list(map(int, true_codes)),
        }
        try:
            r = self.session.post(self.embeddings_url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            self.log.info(f"Sent embeddings | iter={iteration} | shape={embeddings.shape}")
        except requests.RequestException as e:
            self.log.warning(f"Failed to send embeddings @iter={iteration}: {e}")

    def send_annotation_request(self, filenames, indices):
        payload = {"filenames": filenames, "indices": indices}
        try:
            r = self.session.post(self.annotation_url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            self.log.info(f"Sent annotation request | n={len(indices)}")
        except requests.RequestException as e:
            self.log.warning(f"Failed to send annotation request: {e}")

    def send_metrics(
        self,
        iteration,
        accuracy,
        loss,
        accuracy_target=None,
        total_labeled=None,
        human_labeled=None,
        per_class_accuracy=None,
        labeled_counts=None,
        phase=None,
    ):
        payload = {
            "iteration": iteration,
            "accuracy": accuracy,
            "loss": loss,
            "accuracy_target": accuracy_target,
            "total_labeled": total_labeled,
            "human_labeled": human_labeled,
            "per_class_accuracy": per_class_accuracy or {},
            "labeled_counts": labeled_counts or {},
            "phase": phase,
        }
        try:
            r = self.session.post(self.metrics_url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            self.log.info(
                f"Sent metrics | iter={iteration} "
                f"| acc={accuracy:.4f} | loss={loss:.4f} | human_labels={human_labeled}"
            )
        except requests.RequestException as e:
            self.log.warning(f"Failed to send metrics @iter={iteration}: {e}")



