import torch
import torch.nn as nn
import torch.nn.functional as F

class AuxEntTypeTower(nn.Module):
    def __init__(self, hidden_dim, type_size):
        super().__init__()
        self.head = nn.Linear(hidden_dim, type_size)
        self.type_size = type_size
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, feature, target=None, mask=None, weight=None): # weight 沒有用到，忽略即可
        logits = self.head(feature)

        if target is None or mask is None or weight is None:
            return logits, None  # 👉 inference 模式只回傳 logits
    
        logits = torch.masked_select(logits, mask.unsqueeze(-1)).view(-1, self.type_size)
        target = torch.masked_select(target, mask)

        loss = self.criterion(logits, target)

        return logits.detach(), loss
