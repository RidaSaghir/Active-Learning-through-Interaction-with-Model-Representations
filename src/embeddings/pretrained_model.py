import tensorflow_hub as hub
import tensorflow as tf

class YAMNetEmbedder:
    def __init__(self):
        self.model = hub.load("https://tfhub.dev/google/yamnet/1")

    def get_embedding(self, waveform):

        waveform = waveform / tf.int16.max
        waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
        _, embeddings, _ = self.model(waveform)
        return embeddings.numpy().mean(axis=0)
