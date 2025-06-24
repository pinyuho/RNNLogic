import torch
import torch.nn as nn
import torch.nn.functional as F

class MMOEGate(nn.Module):
    """
    shared_feature      : (B, L, D)      ← shared feature
    expert_outputs      : list[(B, L, D)]，長度 = num_experts
    """
    def __init__(self, input_dim, num_experts, normalize="softmax"):
        super().__init__()
        self.num_experts = num_experts
        self.w_gate = nn.Linear(input_dim, num_experts, bias=False)
        self.normalize = normalize

    def forward(self, shared_feature, expert_outputs):
        # 1. 計算 gate raw weights  (B, L, E)
        gate_raw = self.w_gate(shared_feature)

        if self.normalize == "softmax":
            gate = F.softmax(gate_raw, dim=-1)       # 權重和 = 1
        elif self.normalize == "sigmoid":
            gate = torch.sigmoid(gate_raw)           # 權重 ∈ (0,1)
        else:                                        # "none"
            gate = gate_raw                          # 允許任意實數

        # 2. 對 expert_out 做加權和
        #    stack → (B, L, E, D)
        stack = torch.stack(expert_outputs, dim=2)
        mixed_feature = (stack * gate.unsqueeze(-1)).sum(2)  # (B, L, D)

        return mixed_feature, gate          # gate 可拿去觀察 / 可視化
