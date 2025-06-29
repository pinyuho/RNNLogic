import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
import copy
import random
import logging
from collections import defaultdict
from layers import MLP, FuncToNode, FuncToNodeSum
from embedding import RotatE
from torch.utils.data import Dataset, DataLoader
from torch_scatter import scatter, scatter_add, scatter_min, scatter_max, scatter_mean
from typing import Tuple, Optional

class Predictor(torch.nn.Module):
    def __init__(self, graph, gen_rule_accu=None, entity_feature='bias'): # gen_rule_accu: rule * accuracy
        super(Predictor, self).__init__()
        self.graph = graph
        self.num_entities = graph.entity_size
        self.num_relations = graph.relation_size
        self.entity_feature = entity_feature

        self.gen_rule_accu = gen_rule_accu
        if entity_feature == 'bias':
            self.bias = torch.nn.parameter.Parameter(torch.zeros(self.num_entities))

        # self.is_wrnnlogic = is_wrnnlogic
    
    def set_rules(self, input):
        self.rules = list()
        if type(input) == list:
            for rule in input:
                rule_ = (rule[0], rule[1:])
                self.rules.append(rule_)
            logging.info('Predictor: read {} rules from list.'.format(len(self.rules)))
        elif type(input) == str:
            with open(input, 'r') as fi:
                for line in fi:
                    rule = line.strip().split()
                    rule = [int(_) for _ in rule]
                    rule_ = (rule[0], rule[1:])
                    self.rules.append(rule_)
            logging.info('Predictor: read {} rules from file.'.format(len(self.rules)))
        else:
            raise ValueError
        self.num_rules = len(self.rules)

        self.relation2rules = [[] for r in range(self.num_relations)]
        for index, rule in enumerate(self.rules):
            relation = rule[0]
            self.relation2rules[relation].append([index, rule])
        
        self.rule_weights = torch.nn.parameter.Parameter(torch.zeros(self.num_rules))

    def forward(self, all_h, all_r, edges_to_remove):
        query_r = all_r[0].item()
        assert (all_r != query_r).sum() == 0
        device = all_r.device

        score = torch.zeros(all_r.size(0), self.num_entities, device=device)
        mask = torch.zeros(all_r.size(0), self.num_entities, device=device)
        for index, (r_head, r_body) in self.relation2rules[query_r]:
            assert r_head == query_r

            # wRNNLogic
            # x, path_w = self.graph.grounding_with_count(all_h, r_head, r_body, edges_to_remove)
            # score += x * self.rule_weights[index] * path_w

            x, _ = self.graph.grounding(all_h, r_head, r_body, edges_to_remove)
            score += x * self.rule_weights[index]

            mask += x
        
        if mask.sum().item() == 0:
            if self.entity_feature == 'bias':
                return mask + self.bias.unsqueeze(0), (1 - mask).bool()
            else:
                return mask - float('-inf'), mask.bool()
        
        if self.entity_feature == 'bias':
            score = score + self.bias.unsqueeze(0)
            mask = torch.ones_like(mask).bool()
        else:
            mask = (mask != 0)
            score = score.masked_fill(~mask, float('-inf'))
        
        return score, mask

    def compute_H(self, all_h, all_r, all_t, edges_to_remove,
              record_pool=None):

        query_r = all_r[0].item()
        assert (all_r != query_r).sum() == 0          # ✔ 仍保證同 relation
        device,  B = all_r.device, all_h.size(0)

        rule_score, rule_index = [], []
        mask = torch.zeros(B, self.num_entities, device=device)

        # ───────────────────────── Rule loop ──────────────────────────
        for rule_id, (r_head, r_body) in self.relation2rules[query_r]:
            # ↑ ★ 1. 用  `rule_id` 直接拿「規則在全模型中的索引」
            #      不可改成 Python 的 enumerate(idx)，否則和 self.rule_weights
            #      的對應會被打亂！(relation2rules 已保存原本順序)

            assert r_head == query_r

            # ① grounding → 暫存 x_layers
            x_fin, x_layers = self.graph.grounding(
                all_h, r_head, r_body, edges_to_remove,
                keep_layers = record_pool is not None      # ✔
            )

            # ② 回溯路徑（只有 record_pool 時才做）
            if record_pool is not None:
                for b in range(B):
                    # ★ 2. layer[b] 取完就是 1-D Tensor → 保持原寫法
                    path_b = self.graph.recover_path(
                        h_b        = int(all_h[b]),
                        rule_body  = r_body,
                        x_layers_b = [layer[b] for layer in x_layers])

                    if not path_b:
                        continue       # 沒走到尾端 ⇒ 不記

                    hr_idx     = self.graph.encode_hr(all_h[b].item(), r_head)
                    true_tail  = path_b[-1][2]
                    is_correct = int(true_tail in self.graph.hr2o.get(hr_idx, []))

                    record_pool.append({
                        # "rule" : r_body + ["END"],
                        "rule" : r_body,
                        "head" : int(all_h[b]),
                        "path" : path_b,
                        "label": is_correct,
                    })

            # ③ 原來的規則得分
            score = x_fin * self.rule_weights[rule_id]     # ★ 3. 用 rule_id！
            mask  += x_fin
            rule_score.append(score)
            rule_index.append(rule_id)

        # ─────────────────── H-score 計算 ───────────────────
        if not rule_score:                                # relation 沒規則
            return None, None

        rule_index = torch.as_tensor(rule_index, device=device)

        pos_index = F.one_hot(all_t, self.num_entities).bool().to(device)
        neg_index = (mask != 0)

        H_list = []
        for score in rule_score:
            pos = (score * pos_index).sum(1) / pos_index.sum(1).clamp(min=1)
            neg = (score * neg_index).sum(1) / neg_index.sum(1).clamp(min=1)
            H_list.append((pos - neg).unsqueeze(-1))

        H_mat   = torch.cat(H_list, dim=-1)          # (B , #rules_for_r)
        H_final = torch.softmax(H_mat, dim=-1).sum(0)

        return H_final, rule_index


class PredictorPlus(torch.nn.Module):
    def __init__(self, graph, type='emb', num_layers=3, hidden_dim=16, entity_feature='bias', aggregator='sum', embedding_path=None):
        super(PredictorPlus, self).__init__()
        self.graph = graph

        self.type = type
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.entity_feature = entity_feature
        self.aggregator = aggregator
        self.embedding_path = embedding_path

        self.num_entities = graph.entity_size
        self.num_relations = graph.relation_size
        self.padding_index = graph.relation_size

        self.vocab_emb = torch.nn.Embedding(self.num_relations + 1, self.hidden_dim, padding_idx=self.num_relations)
        
        if self.type == 'lstm':
            self.rnn = torch.nn.LSTM(self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True)
        elif self.type == 'gru':
            self.rnn = torch.nn.GRU(self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True)
        elif self.type == 'rnn':
            self.rnn = torch.nn.RNN(self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True)
        elif self.type == 'emb':
            self.rule_emb = None
        else:
            raise NotImplementedError

        if aggregator == 'sum':
            self.rule_to_entity = FuncToNodeSum(self.hidden_dim)
        elif aggregator == 'pna':
            self.rule_to_entity = FuncToNode(self.hidden_dim)
        else:
            raise NotImplementedError

        self.relation_emb = torch.nn.Embedding(self.num_relations, self.hidden_dim)
        self.score_model = MLP(self.hidden_dim * 2, [128, 1]) # 128 for FB15k

        if entity_feature == 'bias':
            self.bias = torch.nn.parameter.Parameter(torch.zeros(self.num_entities))
        elif entity_feature == 'RotatE':
            self.RotatE = RotatE(embedding_path)

    def set_rules(self, input):
        self.rules = list()
        if type(input) == list:
            for rule in input:
                rule_ = (rule[0], rule[1:])
                self.rules.append(rule_)
            logging.info('Predictor+: read {} rules from list.'.format(len(self.rules)))
        elif type(input) == str:
            self.rules = list()
            with open(input, 'r') as fi:
                for line in fi:
                    rule = line.strip().split()
                    rule = [int(_) for _ in rule]
                    rule_ = (rule[0], rule[1:])
                    self.rules.append(rule_)
            logging.info('Predictor+: read {} rules from file.'.format(len(self.rules)))
        else:
            raise ValueError
        self.num_rules = len(self.rules)
        self.max_length = max([len(rule[1]) for rule in self.rules])

        self.relation2rules = [[] for r in range(self.num_relations)]
        for index, rule in enumerate(self.rules):
            relation = rule[0]
            self.relation2rules[relation].append([index, rule])
        
        self.rule_features = []
        for rule in self.rules:
            rule_ = [rule[0]] + rule[1] + [self.padding_index for i in range(self.max_length - len(rule[1]))]
            self.rule_features.append(rule_)
        self.rule_features = torch.tensor(self.rule_features, dtype=torch.long)

        if self.type == 'emb':
            self.rule_emb = nn.parameter.Parameter(torch.zeros(self.num_rules, self.hidden_dim))
            nn.init.kaiming_uniform_(self.rule_emb, a=math.sqrt(5), mode="fan_in")

    def encode_rules(self, rule_features):
        rule_masks = rule_features != self.num_relations
        x = self.vocab_emb(rule_features)
        output, hidden = self.rnn(x)
        idx = (rule_masks.sum(-1) - 1).long()
        idx = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        rule_emb = torch.gather(output, 1, idx).squeeze(1)
        return rule_emb

    def forward(self, all_h, all_r, edges_to_remove):
        query_r = all_r[0].item()
        assert (all_r != query_r).sum() == 0
        device = all_r.device

        if device.type == "cuda":
            self.rule_features = self.rule_features.cuda(device)

        rule_index = list()
        rule_count = list()
        mask = torch.zeros(all_h.size(0), self.graph.entity_size, device=device)
        for index, (r_head, r_body) in self.relation2rules[query_r]:
            assert r_head == query_r

            # wRNNLogic
            # count, _ = self.graph.grounding_with_count(all_h, r_head, r_body, edges_to_remove)
            # count = count.float() 

            count, _ = self.graph.grounding(all_h, r_head, r_body, edges_to_remove)
            count = count.float()
            mask += count

            rule_index.append(index)
            rule_count.append(count)

        if mask.sum().item() == 0:
            if self.entity_feature == 'bias':
                return mask + self.bias.unsqueeze(0), (1 - mask).bool()
            elif self.entity_feature == 'RotatE':
                bias = self.RotatE(all_h, all_r)
                return mask + bias, (1 - mask).bool()
            else:
                return mask - float('-inf'), mask.bool()

        candidate_set = torch.nonzero(mask.view(-1), as_tuple=True)[0]
        batch_id_of_candidate = candidate_set // self.graph.entity_size

        rule_index = torch.tensor(rule_index, dtype=torch.long, device=device)
        rule_count = torch.stack(rule_count, dim=0)
        rule_count = rule_count.reshape(rule_index.size(0), -1)[:, candidate_set]
        
        if self.type == 'emb':
            rule_emb = self.rule_emb[rule_index]
        else:
            rule_emb = self.encode_rules(self.rule_features[rule_index])

        output = self.rule_to_entity(rule_count, rule_emb, batch_id_of_candidate)
        
        rel = self.relation_emb(all_r[0]).unsqueeze(0).expand(output.size(0), -1)
        feature = torch.cat([output, rel], dim=-1)
        output = self.score_model(feature).squeeze(-1)

        score = torch.zeros(all_h.size(0) * self.graph.entity_size, device=device)
        score.scatter_(0, candidate_set, output)
        score = score.view(all_h.size(0), self.graph.entity_size)
        if self.entity_feature == 'bias':
            score = score + self.bias.unsqueeze(0)
            mask = torch.ones_like(mask).bool()
        elif self.entity_feature == 'RotatE':
            bias = self.RotatE(all_h, all_r)
            score = score + bias
            mask = torch.ones_like(mask).bool()
        else:
            mask = (mask != 0)
            score = score.masked_fill(~mask, float('-inf'))

        return score, mask

