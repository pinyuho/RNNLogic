import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
import numpy as np
import logging

class GeneratorMultitask(torch.nn.Module):
    def __init__(self, graph, cluster_size, loss_mode, num_layers, embedding_dim, hidden_dim, emb_init_mode="random"):
        super(GeneratorMultitask, self).__init__()
        self.graph = graph
        self.num_relations = graph.relation_size
        self.loss_mode = loss_mode

        self.num_layers = num_layers
        # self.embedding_dim = embedding_dim
        self.embedding_dim = 512
        self.hidden_dim = hidden_dim

        self.vocab_size = self.num_relations + 2
        self.ending_idx = self.num_relations
        self.padding_idx = self.num_relations + 1

        self.label_size = self.num_relations + 1
        self.cluster_size = cluster_size

        if emb_init_mode == "random":
            self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.padding_idx)
        else: # rotate
            rotate_relation_emb = np.load("../../KnowledgeGraphEmbedding/models/RotatE_semmeddb_512/relation_embedding.npy")
            rotate_relation_emb = torch.tensor(rotate_relation_emb, dtype=torch.float)
            # 隨機初始化額外兩個 embedding
            num_extra = 2
            embedding_dim = rotate_relation_emb.shape[1]
            extra_embeddings = torch.randn(num_extra, embedding_dim) * 0.01  # 小一點較穩定
            # 合併
            full_embedding = torch.cat([rotate_relation_emb, extra_embeddings], dim=0)  # [num_relations + 2, dim]
            self.embedding = torch.nn.Embedding.from_pretrained(full_embedding, freeze=True)

        # ori
        self.encoder = torch.nn.LSTM(self.embedding_dim * 2, self.hidden_dim, self.num_layers, batch_first=True)
        self.mlp_shared = torch.nn.Linear(self.hidden_dim, self.hidden_dim)

        # aux task
        self.mlp_aux = torch.nn.Linear(self.hidden_dim, self.cluster_size)
        self.mlp_aux2 = torch.nn.Linear(self.hidden_dim, self.cluster_size)

        # main task
        self.mlp_main = torch.nn.Linear(self.hidden_dim, self.label_size)

        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.aux_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, inputs, relation, hidden):
        embedding = self.embedding(inputs)
        embedding_r = self.embedding(relation).unsqueeze(1).expand(-1, inputs.size(1), -1)
        embedding = torch.cat([embedding, embedding_r], dim=-1)

        outputs, hidden = self.encoder(embedding, hidden)
        shared = self.mlp_shared(outputs)

        aux_logits = self.mlp_aux(shared) # TODO: 2. 不拿 shared 當做 input, 使用對應的 RotatE relation embedding 當做 input
        aux2_logits = self.mlp_aux2(shared)

        main_logits = self.mlp_main(shared)

        return aux_logits, main_logits, hidden
    
    def get_alpha(self, epoch, max_epoch):
        return min(epoch / max_epoch, 1.0)
        # return 1 / (1 + math.exp(-5 * (epoch / max_epoch - 0.5))) # 更平滑的 sigmoid
    
    def compute_total_loss(self, main_loss, aux_loss, epoch, max_epoch=20):
        if self.loss_mode == 'fixed':
            return 0.8 * main_loss + 0.2 * aux_loss
        
        elif self.loss_mode == 'warmup':
            if epoch > 10:
                return 0.95 * main_loss + 0.05 * aux_loss
            else:
                return main_loss
            
        elif self.loss_mode == 'schedule':
            alpha = min(epoch / max_epoch, 1.0)
            return alpha * main_loss + (1 - alpha) * aux_loss
        
        elif self.loss_mode == 'adaptive':
            total = main_loss + aux_loss + 1e-8
            alpha = main_loss / total
            beta  = aux_loss / total
            return alpha * main_loss + beta * aux_loss
            
        else:
            raise ValueError("Unknown mode.")
        
    # def loss(self, inputs, target, mask, weight, hidden):
    def loss(self, inputs, main_target, aux_target, aux2_target, epoch, mask, weight, hidden):
        aux_logits, main_logits, hidden = self.forward(inputs, inputs[:, 0], hidden)

        main_logits = torch.masked_select(main_logits, mask.unsqueeze(-1)).view(-1, self.label_size)
        main_target = torch.masked_select(main_target, mask)
        main_weight = torch.masked_select((mask.t() * weight).t(), mask)
        main_loss = (self.criterion(main_logits, main_target) * main_weight).sum() / main_weight.sum()

        aux_logits = torch.masked_select(aux_logits, mask.unsqueeze(-1)).view(-1, self.cluster_size)
        aux_target = torch.masked_select(aux_target, mask)
        aux_loss = self.aux_criterion(aux_logits, aux_target)  # 用普通 CrossEntropyLoss 就好，不用用 rule weight 加權

        # aux2_logits = torch.masked_select(aux2_logits, mask.unsqueeze(-1)).view(-1, self.cluster_size)
        # aux2_target = torch.masked_select(aux2_target, mask)
        # aux2_loss = self.aux_criterion(aux2_logits, aux2_target)

        total_loss = self.compute_total_loss(main_loss, aux_loss, epoch) # TODO: aux2_target
        return total_loss, main_loss, aux_loss
