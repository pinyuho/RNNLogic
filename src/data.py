import torch
from torch.utils.data import Dataset
from torch_scatter import scatter
import torch.nn.functional as F

import numpy as np
import os
import random
from easydict import EasyDict
import logging
from typing import List, Optional, Tuple

class KnowledgeGraph(object):
    def __init__(self, data_path):
        self.data_path = data_path

        self.entity2id = dict()
        self.relation2id = dict()
        self.id2entity = dict()
        self.id2relation = dict()
        self.rel2cluster = dict() # auxiliary task

        with open(os.path.join(data_path, 'entities.dict')) as fi:
            for line in fi:
                id, entity = line.strip().split('\t')
                self.entity2id[entity] = int(id)
                self.id2entity[int(id)] = entity

        with open(os.path.join(data_path, 'relations.dict')) as fi:
            for line in fi:
                id, relation = line.strip().split('\t')
                self.relation2id[relation] = int(id)
                self.id2relation[int(id)] = relation

        self.entity_size = len(self.entity2id)
        self.relation_size = len(self.relation2id)
        
        self.train_facts = list()
        self.valid_facts = list()
        self.test_facts = list()
        self.hr2o = dict()
        self.hr2oo = dict()
        self.hr2ooo = dict()
        self.relation2adjacency = [[[], []] for k in range(self.relation_size)]
        self.sparse_adj = {} # To prevent grounding OOM

        self.relation2ht2index = [dict() for k in range(self.relation_size)]
        self.relation2outdegree = [[0 for i in range(self.entity_size)] for k in range(self.relation_size)]

        with open(os.path.join(data_path, "train.txt")) as fi:
            for line in fi:
                h, r, t = line.strip().split('\t')
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                self.train_facts.append((h, r, t))

                hr_index = self.encode_hr(h, r)
                
                if hr_index not in self.hr2o:
                    self.hr2o[hr_index] = list()
                self.hr2o[hr_index].append(t)

                if hr_index not in self.hr2oo:
                    self.hr2oo[hr_index] = list()
                self.hr2oo[hr_index].append(t)

                if hr_index not in self.hr2ooo:
                    self.hr2ooo[hr_index] = list()
                self.hr2ooo[hr_index].append(t)

                self.relation2adjacency[r][0].append(t)
                self.relation2adjacency[r][1].append(h)

                ht_index = self.encode_ht(h, t)
                assert ht_index not in self.relation2ht2index[r]
                index = len(self.relation2ht2index[r])
                self.relation2ht2index[r][ht_index] = index

                self.relation2outdegree[r][t] += 1

        for r, (tails, heads) in enumerate(self.relation2adjacency):
            row = torch.LongTensor(tails)
            col = torch.LongTensor(heads)
            idx = torch.stack([row, col], dim=0)               # shape [2, num_edges]
            vals = torch.ones(row.size(0), dtype=torch.float32)
            A = torch.sparse_coo_tensor(
                idx, vals,
                (self.entity_size, self.entity_size),
                device='cpu'        # 先建在 CPU，之後再 to(device)
            ).coalesce()
            self.sparse_adj[r] = A

        with open(os.path.join(data_path, "valid.txt")) as fi:
            for line in fi:
                h, r, t = line.strip().split('\t')
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                self.valid_facts.append((h, r, t))

                hr_index = self.encode_hr(h, r)

                if hr_index not in self.hr2oo:
                    self.hr2oo[hr_index] = list()
                self.hr2oo[hr_index].append(t)

                if hr_index not in self.hr2ooo:
                    self.hr2ooo[hr_index] = list()
                self.hr2ooo[hr_index].append(t)

        with open(os.path.join(data_path, "test.txt")) as fi:
            for line in fi:
                h, r, t = line.strip().split('\t')
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                self.test_facts.append((h, r, t))

                hr_index = self.encode_hr(h, r)

                if hr_index not in self.hr2ooo:
                    self.hr2ooo[hr_index] = list()
                self.hr2ooo[hr_index].append(t)

        with open(os.path.join(data_path, "relation_cluster.dict")) as fi: # relation_class.dict
            for line in fi:
                rel_id, cluster_id = line.strip().split("\t")
                self.rel2cluster[int(rel_id)] = int(cluster_id)

        for r in range(self.relation_size):
            index = torch.LongTensor(self.relation2adjacency[r])
            value = torch.ones(index.size(1))
            self.relation2adjacency[r] = [index, value]

            self.relation2outdegree[r] = torch.LongTensor(self.relation2outdegree[r])

        print("Data loading | DONE!")

    def encode_hr(self, h, r):
        return r * self.entity_size + h

    def decode_hr(self, index):
        h, r = index % self.entity_size, index // self.entity_size
        return h, r

    def encode_ht(self, h, t):
        return t * self.entity_size + h

    def decode_ht(self, index):
        h, t = index % self.entity_size, index // self.entity_size
        return h, t

    def get_updated_adjacency(self, r, edges_to_remove):
        if edges_to_remove == None:
            return None
        index = self.relation2sparse[r][0]
        value = self.relation2sparse[r][1]
        mask = (index.unsqueeze(1) == edges_to_remove.unsqueeze(-1))
        mask = mask.all(dim=0).any(dim=0)
        mask = ~mask
        index = index[:, mask]
        value = value[mask]
        return [index, value]

    @torch.no_grad()
    def propagate_chunked(self,
                          x: torch.Tensor,         # [E, bs]
                          relation: int,
                          edges_to_remove: Optional[torch.Tensor] = None,
                          chunk_size: int = 32
                         ) -> torch.Tensor:
        """
        把 x ([E,bs]) 切 chunk，分段做 sparse.mm，避免一次 allocate 全 [E,bs] OOM。
        """
        device = x.device
        A = self.sparse_adj[relation].to(device)  # [E, E] sparse
        E, bs = x.shape

        # 預先 allocate 一次 [E, bs]
        out = torch.zeros_like(x)

        # 分段 sparse.mm
        for start in range(0, bs, chunk_size):
            end = min(start + chunk_size, bs)
            out[:, start:end] = torch.sparse.mm(
                A, x[:, start:end]
            )

        # 如果需要剔除邊，就在這裡做 mask
        if edges_to_remove is not None:
            rm = edges_to_remove.view(-1)  # [bs]
            out[rm, torch.arange(bs, device=device)] = 0

        return out
    
    @torch.no_grad()
    def propagate(self,
                  x: torch.Tensor,                    # [E, bs]
                  relation: int,
                  edges_to_remove: Optional[torch.Tensor] = None,
                  chunk_size: int = 32
                 ) -> torch.Tensor:
        """
        Chunked sparse.mm to avoid OOM.
        x:           [E, bs]
        self.sparse_adj[relation]: sparse [E, E]
        """
        device = x.device
        A = self.sparse_adj[relation].to(device)  # sparse adjacency [E, E]
        E, bs = x.shape

        # allocate output once
        out = torch.zeros_like(x)                 # [E, bs]

        # chunked multiplication
        for start in range(0, bs, chunk_size):
            end = min(start + chunk_size, bs)
            out[:, start:end] = torch.sparse.mm(A, x[:, start:end])

        # optional edge removal mask
        if edges_to_remove is not None:
            rm = edges_to_remove.view(-1)         # [bs]
            out[rm, torch.arange(bs, device=device)] = 0

        return out

    @torch.no_grad()
    def grounding(self,
                  h: torch.LongTensor,               # [B]
                  r: int,
                  rule: List[int],
                  edges_to_remove: Optional[torch.Tensor] = None,
                  micro_bs: int = 16,
                  chunk_size: int = 32
                 ) -> torch.Tensor:
        """
        Full grounding with micro-batch + chunked propagation.
        Returns a tensor of shape [B, E].
        """
        device = h.device
        B = h.size(0)
        results = []

        for start in range(0, B, micro_bs):
            end = min(start + micro_bs, B)
            h_mb = h[start:end]                  # [bs]
            bs = end - start

            # one-hot initial distribution [E, bs]
            x = torch.zeros(self.entity_size, bs, device=device)
            x[h_mb, torch.arange(bs, device=device)] = 1.0

            for r_body in rule:
                # slice edges_to_remove for this micro-batch & rule
                if edges_to_remove is not None and r_body == r:
                    e2r_mb = edges_to_remove[start:end].view(-1)
                else:
                    e2r_mb = None

                # chunked sparse propagation
                x = self.propagate(x, r_body, e2r_mb, chunk_size)

            # collect and transpose → [bs, E]
            results.append(x.transpose(0, 1))
            del x

        # concatenate all micro-batches → [B, E]
        return torch.cat(results, dim=0)

class TrainDataset(Dataset):
    def __init__(self, graph, batch_size):
        self.graph = graph
        self.batch_size = batch_size

        self.r2instances = [[] for r in range(self.graph.relation_size)]
        for h, r, t in self.graph.train_facts:
            self.r2instances[r].append((h, r, t))

        self.make_batches()

    def make_batches(self):
        for r in range(self.graph.relation_size):
            random.shuffle(self.r2instances[r])

        self.batches = list()
        for r, instances in enumerate(self.r2instances):
            for k in range(0, len(instances), self.batch_size):
                start = k
                end = min(k + self.batch_size, len(instances))
                self.batches.append(instances[start:end])
        random.shuffle(self.batches)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        data = self.batches[idx]

        all_h = torch.LongTensor([_[0] for _ in data])
        all_r = torch.LongTensor([_[1] for _ in data])
        all_t = torch.LongTensor([_[2] for _ in data])
        target = torch.zeros(len(data), self.graph.entity_size)
        edges_to_remove = []
        for k, (h, r, t) in enumerate(data):
            hr_index = self.graph.encode_hr(h, r)
            t_index = torch.LongTensor(self.graph.hr2o[hr_index])
            target[k][t_index] = 1

            ht_index = self.graph.encode_ht(h, t)
            edge = self.graph.relation2ht2index[r][ht_index]
            edges_to_remove.append(edge)
        edges_to_remove = torch.LongTensor(edges_to_remove)

        return all_h, all_r, all_t, target, edges_to_remove

class ValidDataset(Dataset):
    def __init__(self, graph, batch_size):
        self.graph = graph
        self.batch_size = batch_size

        facts = self.graph.valid_facts

        r2instances = [[] for r in range(self.graph.relation_size)]
        for h, r, t in facts:
            r2instances[r].append((h, r, t))

        self.batches = list()
        for r, instances in enumerate(r2instances):
            random.shuffle(instances)
            for k in range(0, len(instances), self.batch_size):
                start = k
                end = min(k + self.batch_size, len(instances))
                self.batches.append(instances[start:end])

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        data = self.batches[idx]

        all_h = torch.LongTensor([_[0] for _ in data])
        all_r = torch.LongTensor([_[1] for _ in data])
        all_t = torch.LongTensor([_[2] for _ in data])

        mask = torch.ones(len(data), self.graph.entity_size).bool()
        for k, (h, r, t) in enumerate(data):
            hr_index = self.graph.encode_hr(h, r)
            t_index = torch.LongTensor(self.graph.hr2oo[hr_index])
            mask[k][t_index] = 0

        return all_h, all_r, all_t, mask

class TestDataset(Dataset):
    def __init__(self, graph, batch_size):
        self.graph = graph
        self.batch_size = batch_size

        facts = self.graph.test_facts

        r2instances = [[] for r in range(self.graph.relation_size)]
        for h, r, t in facts:
            r2instances[r].append((h, r, t))

        self.batches = list()
        for r, instances in enumerate(r2instances):
            random.shuffle(instances)
            for k in range(0, len(instances), self.batch_size):
                start = k
                end = min(k + self.batch_size, len(instances))
                self.batches.append(instances[start:end])

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        data = self.batches[idx]

        all_h = torch.LongTensor([_[0] for _ in data])
        all_r = torch.LongTensor([_[1] for _ in data])
        all_t = torch.LongTensor([_[2] for _ in data])

        mask = torch.ones(len(data), self.graph.entity_size).bool()
        for k, (h, r, t) in enumerate(data):
            hr_index = self.graph.encode_hr(h, r)
            t_index = torch.LongTensor(self.graph.hr2ooo[hr_index])
            mask[k][t_index] = 0

        return all_h, all_r, all_t, mask

class RuleDataset(Dataset):
    def __init__(self, num_relations, input, rel2cluster=None, cluster_size=None):
        self.rules = list()
        self.num_relations = num_relations
        self.ending_idx = num_relations
        self.padding_idx = num_relations + 1
        
        if type(input) == list:
            rules = input
        elif type(input) == str:
            rules = list()
            with open(input, 'r') as fi:
                for line in fi:
                    rule = line.strip().split()
                    rule = [int(_) for _ in rule[0:-1]] + [float(rule[-1]) * 1000]
                    rules.append(rule)
        
        self.rules = []
        for rule in rules:
            rule_len = len(rule)
            formatted_rule = [rule[0:-1] + [self.ending_idx], self.padding_idx, rule[-1] + 1e-5]
            self.rules.append(formatted_rule)
    
    def __len__(self):
        return len(self.rules)

    def __getitem__(self, idx):
        return self.rules[idx]

    @staticmethod
    def collate_fn(data):
        inputs = [item[0][0:len(item[0])-1] for item in data]
        main_target = [item[0][1:len(item[0])] for item in data]
        weight = [float(item[-1]) for item in data]
        max_len = max([len(_) for _ in inputs])
        padding_index = [int(item[-2]) for item in data]

        # aux_target
        rel2cluster = load_relation_clusters()

        aux_target = [[rel2cluster.get(rel, -1) for rel in seq] for seq in main_target]

        for k in range(len(data)):
            for i in range(max_len - len(inputs[k])):
                inputs[k].append(padding_index[k])
                main_target[k].append(padding_index[k])
                aux_target[k].append(-1)  # -1 代表 padding，避開有效 cluster id 範圍

        inputs = torch.tensor(inputs, dtype=torch.long)
        main_target = torch.tensor(main_target, dtype=torch.long)
        aux_target = torch.tensor(aux_target, dtype=torch.long)
        weight = torch.tensor(weight)
        mask = (main_target != torch.tensor(padding_index, dtype=torch.long).unsqueeze(1))

        # return inputs, target, mask, weight
        return inputs, main_target, aux_target, mask, weight

def load_relation_clusters(data_path="../data/semmed", file_name="relation_cluster.dict"): # relation_class.dict
    rel2cluster = {}
    with open(os.path.join(data_path, file_name)) as fi:
        for line in fi:
            rel_id, cluster_id = line.strip().split("\t")
            rel2cluster[int(rel_id)] = int(cluster_id)
    return rel2cluster

def Iterator(dataloader):
    while True:
        for data in dataloader:
            yield data