import torch
import torch.nn as nn

# ---------- 零件 ----------
from generator_multitask_models.experts.mlp_expert import MLPExpert
from generator_multitask_models.gates.mmoe_gate    import MMOEGate
from generator_multitask_models.towers.next_rel_tower import NextRelTower
from generator_multitask_models.towers.cluster_tower  import ClusterTower
# 若還有其他任務，再在此 import

# ===========================================================================
class MultiTaskMMOE(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        hidden_dim = cfg.hidden_dim

        # 1 ─ Shared Encoder --------------------------------------------------
        self.embedding = nn.Embedding(
            num_embeddings=cfg.vocab_size,
            embedding_dim=cfg.embedding_dim,
            padding_idx=cfg.padding_idx,
        )
        self.lstm = nn.LSTM(
            input_size=cfg.embedding_dim * 2,
            hidden_size=hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
        )

        # 2 ─ Experts ---------------------------------------------------------
        self.experts = nn.ModuleList(
            MLPExpert(in_dim=hidden_dim,
                      hid_dim=hidden_dim,
                      n_layers=cfg.exp_layers,
                      dropout=cfg.exp_dropout)
            for _ in range(cfg.num_experts)
        )

        # 3 ─ Gates（每任務一個 gate）-----------------------------------------
        self.gates = nn.ModuleDict({
            task: MMOEGate(in_dim=hidden_dim, num_experts=cfg.num_experts)
            for task in cfg.tasks
        })

        # 4 ─ Towers（任務專屬頭）---------------------------------------------
        self.towers = nn.ModuleDict({
            "main": NextRelTower (hid_dim=hidden_dim,
                                  vocab_size=cfg.vocab_size,
                                  padding_idx=cfg.padding_idx),
            "aux1": ClusterTower(hid_dim=hidden_dim,
                                  cluster_size=cfg.cluster_size),
            # 例：加入新任務
            # "aux2": NewAuxTower(hid_dim=hid, ...)
        })

    # -----------------------------------------------------------------------
    @torch.no_grad()
    def init_hidden(self, batch_size):
        """LSTM 隱藏態初始化（0 張量即可）"""
        h0 = torch.zeros(self.cfg.num_layers, batch_size, self.cfg.hidden_dim,
                         device=self.embedding.weight.device)
        c0 = torch.zeros_like(h0)
        return h0, c0

    # -----------------------------------------------------------------------
    def forward_shared(self, seq, rel):
        e_seq = self.embedding(seq)                             # (B,L,E)
        e_rel = self.embedding(rel).unsqueeze(1).expand_as(e_seq)
        embed = torch.cat([e_seq, e_rel], dim=-1)               # (B,L,2E)
        out, _ = self.lstm(embed, None)                        # (B,L,H)
        return out

    # -----------------------------------------------------------------------
    def forward(self, batch, task):
        # (1) shared feature
        shared_feat = self.forward_shared(batch["seq"], batch["rel"])

        # (2) experts
        expert_outs = [expert(shared_feat) for expert in self.experts]

        # (3) gate -> 混合特徵
        mixed_feat, gate_w = self.gates[task](shared_feat, expert_outs)

        # (4) tower & loss
        logits, loss = self.towers[task](
            mixed_feat,
            batch[f"{task}_tgt"],
            batch["mask"],
            batch["weight"],
        )
        return logits, loss, gate_w

    # -----------------------------------------------------------------------
    def forward_all(self, batch, loss_weights):
        shared_feat = self.forward_shared(batch["seq"], batch["rel"])
        expert_outs = [expert(shared_feat) for expert in self.experts]

        total_loss, tower_logits = 0.0, {}
        gate_record = {}
        for task, w in loss_weights.items():
            mixed, gate_w = self.gates[task](shared_feat, expert_outs)
            logits, loss = self.towers[task](
                mixed,
                batch[f"{task}_tgt"],
                batch["mask"],
                batch["weight"],
            )
            total_loss += w * loss
            tower_logits[task] = logits
            gate_record[task] = gate_w
        return tower_logits, total_loss, gate_record
