import torch.nn as nn
import torch.nn.functional as F

class AuxRelClusterTower(nn.Module):
    """
    預測「下一個 relation 的 cluster」。
    feature : (B, L, H)
    target  : (B, L)  cluster id，無標註填 -1
    mask    : (B, L)  bool，有效 token
    """
    def __init__(self, hidden_dim, cluster_size):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, cluster_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, feature, target, mask, weight=None):
        logits_full = self.linear(feature)

        logits = logits_full[mask]   # flatten to (N, K)
        target = target[mask]        # (N,)

        # (3) loss（不加權）
        loss = self.criterion(logits, target)

        return logits.detach(), loss
