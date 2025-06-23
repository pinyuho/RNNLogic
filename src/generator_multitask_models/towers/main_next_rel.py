import torch.nn as nn
import torch.nn.functional as F

class MainNextRelTower(nn.Module):
    def __init__(self, hidden_dim, vocab_size, padding_idx):
        super().__init__()
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.padding_idx = padding_idx
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def forward(self, feature, target, mask, weight):
        logits = self.head(feature)              # (B, L, |R|)
        logits = logits[mask]                 # (N, |R|)
        target    = target[mask]                 # (N,)
        weight      = weight[mask]                 # (N,)
        loss   = (self.critetion(logits, target) * weight).sum() / weight.sum()
        return logits.detach(), loss
