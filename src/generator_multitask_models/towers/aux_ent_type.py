import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class AuxEntTypeTower(nn.Module):
    def __init__(self, hidden_dim, type_size, soft_label=False): # soft_label: 機率分佈
        super().__init__()
        self.head = nn.Linear(hidden_dim, type_size)
        self.type_size = type_size

        self.soft_label = soft_label
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.kl_loss  = nn.KLDivLoss(reduction="none")

    def forward(self, feature, target=None, mask=None, weight=None, alpha=0.0): # weight 沒有用到，忽略即可
        # alpha   : 0 → 100% teacher forcing; 1 → 100% 用上一步預測

        # logits = self.head(feature)
        # if target is None or mask is None:
        #     return logits, None  # 👉 inference 模式只回傳 logits
    
        # per_elem_loss = self.criterion(logits, target)       # (B , L , T)
        # mask3d = mask.unsqueeze(-1).float()                  # (B , L , 1)
        # masked_loss = per_elem_loss * mask3d                 # padding → 0
        # loss = masked_loss.sum() / mask3d.sum().clamp(min=1)
        # return logits.detach(), loss

        logits = self.head(feature)
        # ───── 推論階段 ─────
        if target is None or mask is None:
            return logits, None

        B, L, T = logits.shape
        mask3d = mask.unsqueeze(-1).float()               # (B, L, 1)

        # ───── 混合 真實 target 跟 上一部預測值 ─────
        with torch.no_grad():
            prev = target[:, 0]                           # t=0 用真值
            soft_buf = []
            for t in range(L):
                mix = (1 - alpha) * target[:, t] + alpha * prev # 這一行！
                soft_buf.append(mix.unsqueeze(1))         # (B, 1, T)

                # 更新 prev：取當前預測（softmax 或 sigmoid 皆可）
                prev = (
                    F.softmax(logits[:, t], dim=-1) if self.soft_label
                    else torch.sigmoid(logits[:, t])
                ).detach()

            soft_target = torch.cat(soft_buf, dim=1)      # (B, L, T)

        # ───── 選擇 loss ─────
        if self.soft_label:
            # soft-label cross-entropy = − Σ p * log q
            log_q    = F.log_softmax(logits, dim=-1)
            per_elem = -(soft_target * log_q)             # (B, L, T)
        else:
            per_elem = self.bce_loss(logits, soft_target) # (B, L, T)

        loss = (per_elem * mask3d).sum() / mask3d.sum().clamp(min=1.0)
        return logits.detach(), loss