import requests
from config import BROADCAST_INTERVAL

class RestCommunicator:
    def __init__(self, url="http://localhost:8000/embeddings", interval=BROADCAST_INTERVAL):
        self.url = url
        # How often during iterations, the communicator sends embeddings
        self.interval = interval

    def maybe_send(self, iteration, embeddings, labels):
        if iteration % self.interval == 0:
            self.send(iteration, embeddings, labels)

    def send(self, iteration, embeddings, labels):
        payload = {
            "iteration": iteration,
            "embedding_shape": list(embeddings.shape),
            "embeddings": embeddings.tolist(),  # convert to JSON serializable format
            "labels": labels.tolist(),
            "filenames": filenames
        }
        try:
            response = requests.post(self.url, json=payload)
            response.raise_for_status()
            print(f"[RestCommunicator] Sent embeddings at iteration {iteration}")
        except requests.RequestException as e:
            print(f"[RestCommunicator] Error sending data: {e}")
