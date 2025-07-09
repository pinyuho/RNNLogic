import torch

from utils import build_rotate_embedding

from .experts.mlp_expert          import MLPExpert
from .gates.mmoe_gate             import MMOEGate
from .towers.main_next_rel        import MainNextRelTower
from .towers.aux_rel_cluster      import AuxRelClusterTower
from .towers.aux_ent_type         import AuxEntTypeTower

import logging

class MultitaskMMOEAux1(torch.nn.Module):
    def __init__(self, graph, cluster_size, is_soft_label, # is_soft_label: grounding encoding 是否為機率分佈
                 num_layers, embedding_dim, hidden_dim, emb_init_mode="random",
                 num_experts=4, mmoe_dropout=0.1): 
        super().__init__()

        self.graph = graph
        self.num_relations = graph.relation_size
        self.type_size = graph.type_size

        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.vocab_size = self.num_relations + 2
        self.ending_idx = self.num_relations
        self.padding_idx = self.num_relations + 1

        self.label_size = self.num_relations + 1
        self.cluster_size = cluster_size


        if emb_init_mode == "random":
            self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.padding_idx)
        else:  # rotate
            self.embedding = build_rotate_embedding()
            

        self.encoder = torch.nn.LSTM(self.embedding_dim * 2, self.hidden_dim, self.num_layers, batch_first=True)
        self.mlp_shared = torch.nn.Linear(self.hidden_dim, self.hidden_dim)

        # ------------- MMOE ----------------
        self.experts = torch.nn.ModuleList([
            MLPExpert(self.hidden_dim, self.hidden_dim, n_layers=2, dropout=mmoe_dropout)
            for _ in range(num_experts)
        ])

        self.gates = torch.nn.ModuleDict({
            "main":            MMOEGate(self.hidden_dim, num_experts),
            "aux_rel_cluster": MMOEGate(self.hidden_dim, num_experts),
            # "aux_ent_type":    MMOEGate(self.hidden_dim, num_experts),
        })

        self.towers = torch.nn.ModuleDict({
            "main":            MainNextRelTower(self.hidden_dim, self.label_size, self.padding_idx),
            "aux_rel_cluster": AuxRelClusterTower(self.hidden_dim, self.cluster_size),
            # "aux_ent_type":    AuxEntTypeTower(self.hidden_dim, self.type_size, is_soft_label),
        })

    def shared_encode(self, inputs, relation, sem_multihot, hidden):
        
        embedding = self.embedding(inputs)
        embedding_r = self.embedding(relation).unsqueeze(1).expand(-1, inputs.size(1), -1)
        # embedding = torch.cat([sem_multihot, embedding, embedding_r], dim=-1)
        embedding = torch.cat([embedding, embedding_r], dim=-1)

        # logging.info(f"sem: {sem_multihot.shape}")
        # logging.info(embedding.shape)

        outputs, hidden = self.encoder(embedding, hidden)
        shared = self.mlp_shared(outputs)

        return shared, hidden
    
    # def forward(self, inputs, relation, hidden, task="main"):
    def forward(self, batch, hidden, task="main", alpha=0.0):
        shared, _ = self.shared_encode(batch["sequence"], batch["relation"], batch["aux_ent_type_multihot"], hidden)             # (B,L,H)

        # -------- MMOE ----------
        expert_outs = [E(shared) for E in self.experts]            # list len=E
        mixed, _    = self.gates[task](shared, expert_outs)        # (B,L,H)

        # -------- Each task's tower ----------
        extra = {"alpha": alpha} if task == "aux_ent_type" else {}
        if self.training:                               # 訓練模式
            logits, loss = self.towers[task](
                mixed,
                batch[f"{task}_target"],                   # 真 label
                batch["mask"],
                batch["weight"],
                **extra
            )
            return logits, loss, hidden

        else:                                           # 推論 / 評估
            logits, _ = self.towers[task](mixed)        # target=None → 只回 logits
            return logits, None, hidden

    def loss(self, batch, hidden, task="main", alpha=0.0):
        logits, loss, hidden = self.forward(batch, hidden, task, alpha)

        # 在 train 模式下 forward 會算好 loss；eval 模式 loss= None
        # 這裡只在訓練階段被呼叫，所以直接回傳
        return loss