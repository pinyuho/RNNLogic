import torch
import torch.nn as nn

from utils import build_rotate_embedding

from .experts.mlp_expert          import MLPExpert
from .gates.mmoe_gate             import MMOEGate
from .towers.main_next_rel        import MainNextRelTower
from .towers.aux_rel_cluster      import AuxRelClusterTower
from .towers.aux_ent_type         import AuxEntTypeTower

class MultitaskMMOE(nn.Module):
    def __init__(self, graph, cluster_size, loss_mode, num_layers, embedding_dim, hidden_dim, emb_init_mode="random",
                 num_experts=4, mmoe_dropout=0.1): 
        super().__init__()

        self.graph = graph
        self.num_relations = graph.relation_size
        self.loss_mode = loss_mode

        self.num_layers = num_layers
        self.embedding_dim = 512        # 仍固定 512, FIXME: flexible
        self.hidden_dim = hidden_dim

        self.vocab_size = self.num_relations + 2
        self.ending_idx = self.num_relations
        self.padding_idx = self.num_relations + 1

        self.label_size = self.num_relations + 1
        self.cluster_size = cluster_size

        self.type_size = 136 # FIXME: hard coded

        self.encoder = nn.LSTM(self.embedding_dim * 2, self.hidden_dim, self.num_layers, batch_first=True)

        if emb_init_mode == "random":
            self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim,
                                          padding_idx=self.padding_idx)
        else:  # rotate
            self.embedding = build_rotate_embedding(
                "../../KnowledgeGraphEmbedding/models/RotatE_semmeddb_512/relation_embedding.npy"
            )

        self.encoder = torch.nn.LSTM(self.embedding_dim * 2, self.hidden_dim, self.num_layers, batch_first=True)
        self.mlp_shared = torch.nn.Linear(self.hidden_dim, self.hidden_dim)

        # ------------- MMOE ----------------
        self.experts = nn.ModuleList([
            MLPExpert(self.hidden_dim, self.hidden_dim, n_layers=2, dropout=mmoe_dropout)
            for _ in range(num_experts)
        ])

        self.gates = nn.ModuleDict({
            "main":            MMOEGate(self.hidden_dim, num_experts),
            "aux_rel_cluster": MMOEGate(self.hidden_dim, num_experts),
            "aux_ent_type":    MMOEGate(self.hidden_dim, num_experts),
        })

        self.towers = nn.ModuleDict({
            "main":            MainNextRelTower(self.hidden_dim, self.label_size, self.padding_idx),
            "aux_rel_cluster": AuxRelClusterTower(self.hidden_dim, self.cluster_size),
            "aux_ent_type":    AuxEntTypeTower(self.hidden_dim, self.type_size),
        })

        self.main_criterion  = torch.nn.CrossEntropyLoss(reduction='none') # reduction 自訂權重用的，其他 task 沒用到 weight 則不用
        self.aux1_criterion  = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.aux2_criterion  = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def shared_encode(self, inputs, relation, hidden):
        
        embedding = self.embedding(inputs)
        embedding_r = self.embedding(relation).unsqueeze(1).expand(-1, inputs.size(1), -1)
        embedding = torch.cat([embedding, embedding_r], dim=-1)

        outputs, hidden = self.encoder(embedding, hidden)
        shared = self.mlp_shared(outputs)

        return shared, hidden
    
    # def forward(self, inputs, relation, hidden, task="main"):
    def forward(self, batch, hidden, task="main"):
        shared, _ = self.shared_encode(batch["sequence"], batch["relation"], hidden)             # (B,L,H)

        # -------- MMOE ----------
        expert_outs = [E(shared) for E in self.experts]            # list len=E
        mixed, _    = self.gates[task](shared, expert_outs)        # (B,L,H)

        # -------- Each task's tower ----------
        if self.training:                               # 訓練模式
            logits, loss = self.towers[task](
                mixed,
                batch[f"{task}_target"],                   # 真 label
                batch["mask"],
                batch["weight"]
            )
            return logits, loss, hidden

        else:                                           # 推論 / 評估
            logits, _ = self.towers[task](mixed)        # target=None → 只回 logits
            return logits, None, hidden

    def loss(self, batch, hidden, task="main"): # TODO: 這個是舊的 multitask 的 loss
        logits, loss, hidden = self.forward(batch, hidden, task)

        # 在 train 模式下 forward 會算好 loss；eval 模式 loss= None
        # 這裡只在訓練階段被呼叫，所以直接回傳
        return loss