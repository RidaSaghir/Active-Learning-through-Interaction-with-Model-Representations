import torch
import torch.nn as nn
import torch.nn.functional as F
from config import LEARNING_RATE

class TrainableModel(nn.Module):
    def __init__(self, input_dim=1024, embedding_dim=64, num_classes=10, lr=LEARNING_RATE):
        super().__init__()
        self.embedding_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)  # Learnable embedding for VR
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)
        z = self.embedding_layer(x)   # z is the dynamic, learnable embedding
        logits = self.classifier(z)   # classification head
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
