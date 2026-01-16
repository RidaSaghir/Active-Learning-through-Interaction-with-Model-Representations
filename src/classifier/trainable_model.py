import torch
import torch.nn as nn
import torch.nn.functional as F
from config import LEARNING_RATE

class TrainableModel(nn.Module):
    def __init__(self, projector, classifier, lr):
        super().__init__()
        self.projector = projector
        self.classifier = classifier
        self.optimizer = (
            torch.optim.Adam(self.parameters(), lr=lr)
            if isinstance(classifier, nn.Module)
            else None
        )

    def forward(self, x):
        z = self.projector(x)
        logits = self.classifier(z)
        return logits, z


    def train_step(self, x, y):
        self.train()
        logits, _ = self.forward(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        correct = (preds == y).sum().item()
        total = y.size(0)
        accuracy = correct / total

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), accuracy

class ProjectionHead(nn.Module):
    def forward(self, x):
        raise NotImplementedError

class MLPProjector(ProjectionHead):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class LinearProjector(ProjectionHead):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

class IdentityProjector(ProjectionHead):
    def forward(self, x):
        return x

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, z):
        return self.fc(z)

class SklearnClassifier:
    def __init__(self, clf):
        self.clf = clf

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
