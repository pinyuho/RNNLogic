import torch
import torch.nn as nn

from utils import build_rotate_embedding

from .towers.main_next_rel        import MainNextRelTower
from .towers.aux_rel_cluster      import AuxRelClusterTower
from .towers.aux_ent_type         import AuxEntTypeTower

class MultitaskHardSharing(nn.Module):
    def __init__(self, graph, cluster_size, num_layers, embedding_dim, hidden_dim, emb_init_mode="random"): 
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

        self.encoder = nn.LSTM(self.embedding_dim * 2, self.hidden_dim, self.num_layers, batch_first=True)

        if emb_init_mode == "random":
            self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim,
                                          padding_idx=self.padding_idx)
        else:  # rotate
            self.embedding = build_rotate_embedding()

        self.encoder = torch.nn.LSTM(self.embedding_dim * 2, self.hidden_dim, self.num_layers, batch_first=True)
        self.mlp_shared = torch.nn.Linear(self.hidden_dim, self.hidden_dim)

        self.towers = nn.ModuleDict({
            "main":            MainNextRelTower(self.hidden_dim, self.label_size, self.padding_idx),
            # "aux_rel_cluster": AuxRelClusterTower(self.hidden_dim, self.cluster_size),
            "aux_ent_type":    AuxEntTypeTower(self.hidden_dim, self.type_size),
        })

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

        # -------- Each task's tower ----------
        if self.training:                               # 訓練模式
            logits, loss = self.towers[task](
                shared,
                batch[f"{task}_target"],                   # 真 label
                batch["mask"],
                batch["weight"]
            )
            return logits, loss, hidden

        else:                                           # 推論 / 評估
            logits, _ = self.towers[task](shared)        # target=None → 只回 logits
            return logits, None, hidden

    def loss(self, batch, hidden, task="main"): # TODO: 這個是舊的 multitask 的 loss
        logits, loss, hidden = self.forward(batch, hidden, task)

        # 在 train 模式下 forward 會算好 loss；eval 模式 loss= None
        # 這裡只在訓練階段被呼叫，所以直接回傳
        return loss