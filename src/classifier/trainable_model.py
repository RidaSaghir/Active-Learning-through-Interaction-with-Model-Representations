import torch
import torch.nn as nn
import torch.nn.functional as F

class TrainableModel(nn.Module):
    def __init__(self, input_dim=1024, embedding_dim=64, num_classes=10, lr=1e-3):
        super().__init__()
        self.embedding_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)  # Learnable embedding for VR
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        z = self.embedding_layer(x)   # z is the dynamic, learnable embedding
        logits = self.classifier(z)   # classification head
        return logits, z

    def train_step(self, x, y):
        self.train()
        logits, _ = self.forward(x)
        loss = F.cross_entropy(logits, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
