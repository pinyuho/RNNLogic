import torch
import torch.nn as nn
import torch.nn.functional as F

class MainNextRelTower(nn.Module):
    def __init__(self, hidden_dim, label_size, padding_idx): # label_size = relation_size + 1 (ending token)
        super().__init__()
        self.head = nn.Linear(hidden_dim, label_size)
        self.label_size = label_size
        self.padding_idx = padding_idx
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def forward(self, feature, target, mask, weight):
        logits = self.head(feature)
        logits = torch.masked_select(logits, mask.unsqueeze(-1)).view(-1, self.label_size)
        target = torch.masked_select(target, mask)
        weight = torch.masked_select((mask.t() * weight).t(), mask)
        loss = (self.criterion(logits, target) * weight).sum() / weight.sum()

        return logits.detach(), loss
