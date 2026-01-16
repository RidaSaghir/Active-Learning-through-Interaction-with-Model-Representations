import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_model import BaseModel
from config import LEARNING_RATE

class TorchModel(BaseModel, nn.Module):
    def __init__(self, projector, classifier, num_classes):
        super().__init__()
        self.projector = projector
        self.classifier = classifier
        self.num_classes = num_classes
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

    def forward(self, x):
        z = self.projector(x)
        logits = self.classifier(z)
        return logits, z

    def train_step(self, x, y):
        self.train()
        logits, _ = self.forward(x)
        loss = F.cross_entropy(logits, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        acc = (logits.argmax(dim=1) == y).float().mean().item()
        return loss.item(), acc

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            return torch.softmax(logits, dim=1).cpu().numpy()

    def project(self, x):
        self.eval()
        with torch.no_grad():
            z = self.projector(x)
            return z.cpu().numpy()

class MLPProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class LinearProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)
