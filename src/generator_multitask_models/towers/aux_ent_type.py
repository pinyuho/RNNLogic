# towers/aux_ent_type.py
import torch.nn as nn
import torch.nn.functional as F

class AuxEntTypeTower(nn.Module):
    """
    預測「下一個 entity 的語意 / type」(例如 WordNet lexname)。
    feature : (B, L, H)
    target  : (B, L)  type id，無標註填 -1
    mask    : (B, L)  bool，有效 token
    """
    def __init__(self, hidden_dim, num_type):
        super().__init__()
        self.head = nn.Linear(hidden_dim, num_type)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, feature, target, mask, weight=None):
        logits_full = self.head(feature)

        logits = logits_full[mask]   # (N, T)
        target = target[mask]        # (N,)

        loss = self.criterion(logits, target)

        return logits.detach(), loss
