import torch
from torch.utils.data import Dataset
from functools import partial

def load_relation_clusters(relation_cluster_file):
    rel2cluster = {}
    with open(relation_cluster_file) as fi:
        for line in fi:
            rel_id, cluster_id = line.strip().split("\t")
            rel2cluster[int(rel_id)] = int(cluster_id)
    return rel2cluster

class RuleDataset(Dataset):
    def __init__(self, num_relations, input, cluster_size, relation_cluster_file, is_wrnnlogic=False):
        self.rules = list()
        self.num_relations = num_relations
        self.ending_idx = num_relations
        self.padding_idx = num_relations + 1

        self.cluster_size = cluster_size
        self.rel2cluster = load_relation_clusters(relation_cluster_file)

        self.collate_fn = partial(self._collate_static, self.rel2cluster)

        self.rule_accuracies = list()
        
        if isinstance(input, list):
            rules_flat = input                          # sample() 回傳的那種
        elif isinstance(input, str):
            rules_flat = []
            with open(input, 'r') as fi:
                for line in fi:
                    parts = line.strip().split()
                    # [head, body..., conf*1000]
                    rule = [int(x) for x in parts[:-1]] + [float(parts[-1])*1000]
                    rules_flat.append(rule)
        else:
            raise ValueError("input 必須是 list 或檔案路徑")

        # 把「扁平格式」保留下來，方便第一輪 RP 直接使用
        self.rp_input = rules_flat

        # ───────── 轉成三段式 (原本流程) ─────────
        for rule in rules_flat:
            conf = rule[-1]
            rels = rule[:-1] + [self.ending_idx]
            formatted_rule = [rels, self.padding_idx, conf + 1e-5]
            self.rules.append(formatted_rule)
    
    def __len__(self):
        return len(self.rules)

    def __getitem__(self, idx):
        return self.rules[idx]
    
    @staticmethod
    def _collate_static(rel2cluster, data):
        inputs = [item[0][0:len(item[0])-1] for item in data]
        main_target = [item[0][1:len(item[0])] for item in data]

        weight = [float(item[-1]) for item in data]
        max_len = max([len(_) for _ in inputs])
        padding_index = [int(item[-2]) for item in data]

        aux_rel_cluster_target = [[rel2cluster.get(r, -1) for r in seq] for seq in main_target]
        aux_ent_type_target = [[rel2cluster.get(r, -1) for r in seq] for seq in main_target] # TODO:

        for k in range(len(data)):
            for i in range(max_len - len(inputs[k])):
                inputs[k].append(padding_index[k])
                main_target[k].append(padding_index[k])
                aux_rel_cluster_target[k].append(-1)  # -1 代表 padding，避開有效 cluster id 範圍, CrossEntropy 要搭配設定 ignore index = -1
                aux_ent_type_target[k].append(-1)     # -1 代表 padding，避開有效 cluster id 範圍, CrossEntropy 要搭配設定 ignore index = -1

        inputs = torch.tensor(inputs, dtype=torch.long)
        main_target = torch.tensor(main_target, dtype=torch.long)
        aux_rel_cluster_target = torch.tensor(aux_rel_cluster_target, dtype=torch.long)
        aux_ent_type_target = torch.tensor(aux_ent_type_target, dtype=torch.long)
        weight = torch.tensor(weight)
        mask = (main_target != torch.tensor(padding_index, dtype=torch.long).unsqueeze(1))

        # return inputs, main_target, aux_target, aux_ent_type_target, mask, weight
        batch = {
            "sequence":               inputs,                           # rule sequence
            "relation":               inputs[:, 0],                     # rule head (第一個 token)
            "mask":                   mask,
            "weight":                 weight,
            "main_target":            main_target,
            "aux_rel_cluster_target": aux_rel_cluster_target,
            "aux_ent_type_target":    aux_ent_type_target,
        }
        return batch

