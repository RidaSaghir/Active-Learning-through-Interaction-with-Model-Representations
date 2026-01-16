import numpy as np
from classifier.base_model import BaseModel

class SklearnModel(BaseModel):
    def __init__(self, classifier, projector):
        self.clf = classifier
        self.projector = projector
        self._is_fitted = False

    def train_step(self, x, y):
        # x: torch tensor → numpy
        X = self.projector(x).cpu().numpy()
        y = y.cpu().numpy()
        self.clf.fit(X, y)
        self._is_fitted = True
        return 0.0, 0.0

    def predict_proba(self, x):
        if not self._is_fitted:
            n = x.shape[0]
            return np.ones((n, self.clf.n_classes_)) / self.clf.n_classes_

        X = self.projector(x).cpu().numpy()
        return self.clf.predict_proba(X)

    def project(self, x):
        return self.projector(x).cpu().numpy()

    def eval(self):
        pass
