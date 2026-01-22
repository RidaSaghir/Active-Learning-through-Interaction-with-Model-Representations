import numpy as np
from classifier.base_model import BaseModel
import torch
import random
import joblib
from config import SEED

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class SklearnModel(BaseModel):
    def __init__(self, classifier, projector):
        self.clf = classifier
        self.projector = projector
        self._is_fitted = False
        # Freeze projector explicitly
        for p in self.projector.parameters():
            p.requires_grad = False
        self.projector.eval()

    @property
    def supports_batch_training(self):
        return False

    def fit_full(self, X, y):
        self.clf.fit(X, y)
        self._is_fitted = True

    def train_step(self, x, y):
        raise RuntimeError("train_step() should not be called sklearn SVM / LogReg")

    def predict_proba(self, x):
        if not self._is_fitted:
            n = x.shape[0]
            return np.ones((n, self.clf.n_classes_)) / self.clf.n_classes_

        X = x.detach().cpu().numpy()
        return self.clf.predict_proba(X)

    def save(self, path):
        joblib.dump({
            "classifier": self.clf,
            "is_fitted": self._is_fitted,
        }, path)

    def project_for_view(self, x):
        with torch.no_grad():
            return self.projector(x).cpu().numpy()

    def project(self, x):
        """
        Generic projection hook required by BaseModel.
        For sklearn models, this is identical to project_for_view().
        """
        with torch.no_grad():
            return self.projector(x).cpu().numpy()


    def load(self, path):
        data = joblib.load(path)
        self.clf = data["classifier"]
        self._is_fitted = data.get("is_fitted", True)


    def eval(self):
        pass
