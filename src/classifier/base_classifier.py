import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, optimizer_cls=torch.optim.Adam, lr=1e-3):
        super(AudioClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        self.optimizer = optimizer_cls(self.parameters(), lr=lr)

    def forward(self, x):
        return self.net(x)

    def train_step(self, x, y):
        self.train()
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()