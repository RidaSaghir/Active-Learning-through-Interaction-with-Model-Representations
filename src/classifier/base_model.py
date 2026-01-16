from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    @abstractmethod
    def train_step(self, x, y):
        """One training step (or batch fit)."""
        pass

    @abstractmethod
    def predict_proba(self, x) -> np.ndarray:
        """Return [B, C] probabilities."""
        pass

    @abstractmethod
    def project(self, x) -> np.ndarray:
        """Return low-D embedding [B, d]."""
        pass

    @abstractmethod
    def eval(self):
        pass
