import torch
import numpy as np

def compute_total_loss(
        self,
        main_loss: torch.Tensor,
        aux_losses: list[torch.Tensor],
        epoch: int,
        max_epoch: int = 20,
        warmup_epochs: int = 10,
):
    n_aux = len(aux_losses)
    if n_aux == 0:
        return main_loss

    if self.loss_mode == "fixed":
        return 0.8 * main_loss + 0.1 * aux_losses[0] + 0.1 * aux_losses[1]
        # return 0.6 * main_loss + 0.2 * aux_losses[0] + 0.2 * aux_losses[1]
        # return 0.6 * main_loss + 0.2 * aux_losses[0] + 0.2 * aux_losses[1]
        # return (1/3) * main_loss + (1/3) * aux_losses[0] + (1/3) * aux_losses[1]


    elif self.loss_mode == 'warmup':
        if epoch < warmup_epochs:
            return main_loss
        aux_w_each = 0.05 / n_aux      # 例：總共 0.05
        main_w = 0.95
        aux_part = sum(aux_w_each * l for l in aux_losses)
        return main_w * main_loss + aux_part

    elif self.loss_mode == 'schedule':
        alpha = min(epoch / max_epoch, 1.0)   # alpha 越接近 1 → main 比重越大
        aux_w_each = (1 - alpha) / n_aux
        aux_part = sum(aux_w_each * l for l in aux_losses)
        return alpha * main_loss + aux_part

    elif self.loss_mode == 'adaptive':
        all_losses = torch.stack([main_loss] + aux_losses)
        weights = all_losses / (all_losses.sum() + 1e-8)  # 每項占比
        return (weights[0] * main_loss +
                sum(w * l for w, l in zip(weights[1:], aux_losses)))

    else:
        raise ValueError(f"Unknown loss_mode: {self.loss_mode}")

def build_rotate_embedding(num_extra=2, scale=0.01):
    # path = "../../KnowledgeGraphEmbedding/models/RotatE_semmeddb_0426_all_256/relation_embedding.npy"
    path = "../../KnowledgeGraphEmbedding/models/RotatE_semmeddb_512/relation_embedding.npy"
    """
    讀 RotatE 向量，再補 2 格隨機權重 ⇒ nn.Embedding
    """
    base = torch.as_tensor(np.load(path), dtype=torch.float)          # (R, D)
    extra = torch.randn(num_extra, base.size(1)).mul_(scale)          # (2, D)
    weight = torch.cat([base, extra], dim=0)                          # (R+2, D)
    return torch.nn.Embedding.from_pretrained(
        weight, freeze=True, padding_idx=weight.size(0)-1             # 最後一格當 padding
    )