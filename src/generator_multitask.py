import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report

class GeneratorMultitask(torch.nn.Module):
    def __init__(self, graph, cluster_size, loss_mode, num_layers, embedding_dim, hidden_dim):
        super(GeneratorMultitask, self).__init__()
        self.graph = graph
        self.num_relations = graph.relation_size
        self.loss_mode = loss_mode

        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.vocab_size = self.num_relations + 2
        self.ending_idx = self.num_relations
        self.padding_idx = self.num_relations + 1

        self.label_size = self.num_relations + 1
        self.cluster_size = cluster_size

        # prepare relations' embs
        self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.padding_idx)

        # ori
        self.encoder = torch.nn.LSTM(self.embedding_dim * 2, self.hidden_dim, self.num_layers, batch_first=True)
        self.mlp1 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)

        # aux task
        self.mlp2 = torch.nn.Linear(self.hidden_dim, self.cluster_size)

        # main task
        self.mlp3 = torch.nn.Linear(self.hidden_dim, self.label_size)

        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.aux_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, inputs, relation, hidden):
        embedding = self.embedding(inputs)
        embedding_r = self.embedding(relation).unsqueeze(1).expand(-1, inputs.size(1), -1)
        embedding = torch.cat([embedding, embedding_r], dim=-1)

        outputs, hidden = self.encoder(embedding, hidden)
        shared = self.mlp1(outputs)

        aux_logits = self.mlp2(shared)
        main_logits = self.mlp3(shared)

        return aux_logits, main_logits, hidden
    
    def get_alpha(self, epoch, max_epoch):
        return min(epoch / max_epoch, 1.0)
        # return 1 / (1 + math.exp(-5 * (epoch / max_epoch - 0.5))) # 更平滑的 sigmoid
    
    def compute_total_loss(self, main_loss, aux_loss, topk_loss, epoch, max_epoch=20):
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
    def loss(self, inputs, main_target, aux_target, epoch, mask, weight, hidden):
        aux_logits, main_logits, hidden = self.forward(inputs, inputs[:, 0], hidden)

        main_logits = torch.masked_select(main_logits, mask.unsqueeze(-1)).view(-1, self.label_size)
        main_target = torch.masked_select(main_target, mask)
        main_weight = torch.masked_select((mask.t() * weight).t(), mask)
        main_loss = (self.criterion(main_logits, main_target) * main_weight).sum() / main_weight.sum()

        aux_logits = torch.masked_select(aux_logits, mask.unsqueeze(-1)).view(-1, self.cluster_size)
        aux_target = torch.masked_select(aux_target, mask)
        aux_loss = self.aux_criterion(aux_logits, aux_target)  # 用普通 CrossEntropyLoss 就好，不用用 rule weight 加權

        total_loss = self.compute_total_loss(main_loss, aux_loss, None, epoch)
        # return total_loss
        return total_loss, main_loss, aux_loss# 新增 main_loss, aux_loss for logging

    
        # Accuracy
        # with torch.no_grad():
        #     aux_pred = aux_logits.argmax(dim=-1)
        #     valid = aux_target != -1
        #     aux_acc = (aux_pred[valid] == aux_target[valid]).sum().item() / valid.sum().item()
        #     print("aux_acc: ", aux_acc)