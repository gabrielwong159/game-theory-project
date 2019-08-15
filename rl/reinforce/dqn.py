import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_hidden=128):
        torch.manual_seed(0)
        super().__init__()

        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_outputs)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        logits = self.fc3(x)
        probs = self.softmax(logits)
        
        probs = probs.clamp(1e-3, 1.0)
        probs /= probs.sum()
        return probs
