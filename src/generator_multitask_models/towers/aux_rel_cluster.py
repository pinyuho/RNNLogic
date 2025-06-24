import torch
import torch.nn as nn
import torch.nn.functional as F

class AuxRelClusterTower(nn.Module):
    def __init__(self, hidden_dim, cluster_size):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, cluster_size)
        self.cluster_size = cluster_size
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, feature, target=None, mask=None, weight=None):
        logits= self.linear(feature)

        if target is None or mask is None or weight is None:
            return logits, None  # 👉 inference 模式只回傳 logits

        logits = torch.masked_select(logits, mask.unsqueeze(-1)).view(-1, self.cluster_size)
        target = torch.masked_select(target, mask)

        loss = self.criterion(logits, target)

        return logits.detach(), loss
