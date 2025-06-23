import torch
from torch_scatter import scatter
import os
import logging
import comm

class KnowledgeGraph(object):
    def __init__(self, data_path):
        self.data_path = data_path

        self.entity2id = dict()
        self.relation2id = dict()
        self.id2entity = dict()
        self.id2relation = dict()
        # self.rel2cluster = dict() # auxiliary task

        seen = set()
        with open(os.path.join(data_path, 'entities.dict')) as fi:
            for line in fi:
                id, entity = line.strip().split('\t')
                self.entity2id[entity] = int(id)
                self.id2entity[int(id)] = entity
                if entity in seen:
                    print(f"Duplicate entity name detected: {entity}")
                seen.add(entity)

        with open(os.path.join(data_path, 'relations.dict')) as fi:
            for line in fi:
                id, relation = line.strip().split('\t')
                self.relation2id[relation] = int(id)
                self.id2relation[int(id)] = relation

        self.entity_size = len(self.entity2id)
        self.relation_size = len(self.relation2id)

        print("entity size: ", self.entity_size)
        print("relation size: ", self.relation_size)
        
        self.train_facts = list()
        self.train_counts = list()

        self.valid_facts = list()
        self.test_facts = list()
        self.hr2o = dict()
        self.hr2oo = dict()
        self.hr2ooo = dict()
        self.relation2adjacency = [[[], []] for k in range(self.relation_size)]
        self.relation2ht2index = [dict() for k in range(self.relation_size)]
        self.relation2outdegree = [[0 for i in range(self.entity_size)] for k in range(self.relation_size)]

        self.triple2count = dict() # train dataset
        self.default_cnt = 1.0 # default count for unseen triples
        

        with open(os.path.join(data_path, "train_filtered_with_count.txt")) as fi:
            for line in fi:
                h, r, t, count = line.strip().split('\t')
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                self.train_facts.append((h, r, t))
                
                self.triple2count[(h, r, t)] = int(count)

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

                try:
                    self.relation2outdegree[r][t] += 1
                except:
                    print(f"Error: {r} {h} {t}")
                    raise

        with open(os.path.join(data_path, "valid_filtered.txt")) as fi:
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

        with open(os.path.join(data_path, "test_filtered.txt")) as fi:
        # with open(os.path.join(data_path, "test.txt")) as fi:
            for line in fi:
                h, r, t = line.strip().split('\t')
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                self.test_facts.append((h, r, t))

                hr_index = self.encode_hr(h, r)

                if hr_index not in self.hr2ooo:
                    self.hr2ooo[hr_index] = list()
                self.hr2ooo[hr_index].append(t)


        self.relation2edgecount = []
        for r, adjs in enumerate(self.relation2adjacency): # 每個 relation 都有一個 [[], []]，第一個是 tail index list，第二個是 head index list
            # logging.info(adjs)
            tails_list = adjs[0]     # tail entity idx
            heads_list = adjs[1]     # head entity idx

            edge_cnt = []
            for h_idx, t_idx in zip(heads_list, tails_list):              # 順序保持不變
                h = self.id2entity[h_idx]   # ★ 先變 int
                t = self.id2entity[t_idx]
                c = self.triple2count.get((h, r, t), self.default_cnt)
                edge_cnt.append(c)
            self.relation2edgecount.append(torch.tensor(edge_cnt, dtype=torch.float)) # r -> [(h1, r, t1 這個 pair 的 count), (h2, r, t2 的 count), ...]


        for r in range(self.relation_size):
            index = torch.LongTensor(self.relation2adjacency[r])
            value = torch.ones(index.size(1))
            self.relation2adjacency[r] = [index, value]

            self.relation2outdegree[r] = torch.LongTensor(self.relation2outdegree[r])

        if comm.get_rank() == 0:
            logging.info("Data loading | DONE!")

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

    def grounding_with_count(self, h, r, rule, edges_to_remove):
        device = h.device
        B = h.size(0)
        with torch.no_grad():
            x = torch.nn.functional.one_hot(h, self.entity_size).transpose(0, 1).unsqueeze(-1).cuda(device)

            min_cnt = torch.full((self.entity_size, B), float("inf"), device=device)
            min_cnt[h, torch.arange(B, device=device)] = 0.0     # 起點 0

            for r_body in rule: 
                if r_body == r:
                    x, min_cnt = self.propagate_with_count(x, min_cnt, r_body, edges_to_remove)
                else:
                    x, min_cnt = self.propagate_with_count(x, min_cnt, r_body, None)

        x = x.squeeze(-1).T               # 原本的輸出
        path_w = min_cnt.T                # (B,E) —— 路徑最小 triple-count
        return x, path_w

    def grounding(self, h, r, rule, edges_to_remove):
        device = h.device
        with torch.no_grad():
            x = torch.nn.functional.one_hot(h, self.entity_size).transpose(0, 1).unsqueeze(-1).cuda(device)

            for r_body in rule:
                if r_body == r:
                    x = self.propagate(x, r_body, edges_to_remove)
                else:
                    x = self.propagate(x, r_body, None)

        return x.squeeze(-1).transpose(0, 1)

    
    def propagate_with_count(self, x, min_cnt, relation, edges_to_remove=None): # recursive propagation
        device = x.device

        node_in = self.relation2adjacency[relation][0][1]
        node_out = self.relation2adjacency[relation][0][0]

        # heads = self.relation2adjacency[relation][0][1]   # head index list
        # tails = self.relation2adjacency[relation][0][0]   # tail index list

        edge_cnt = self.relation2edgecount[relation].to(x.device)  # (edge,)

        if device.type == "cuda":
            node_in = node_in.cuda(device)
            node_out = node_out.cuda(device)
            edge_cnt = edge_cnt.cuda(device)

        message = x[node_in]
        cnt_msg  = min_cnt[node_in]
        E, B, D = message.size()

        cnt_msg = torch.minimum(cnt_msg, edge_cnt.unsqueeze(1))  # (edge, B)

        if edges_to_remove == None:
            x = scatter(message, node_out, dim=0, dim_size=x.size(0))
            min_cnt = scatter(cnt_msg, node_out, dim=0, dim_size=min_cnt.size(0), reduce="min")     

        else:
            # message: edge * batch * dim
            message = message.view(-1, D)
            cnt_msg = cnt_msg.view(-1)   # 把 (edge,B) 攤平成 (edge*B,)

            bias = torch.arange(B)
            if device.type == "cuda":
                bias = bias.cuda(device)

            edges_to_remove = edges_to_remove * B + bias

            message[edges_to_remove] = 0
            cnt_msg[edges_to_remove] = float("inf")

            message = message.view(E, B, D)
            cnt_msg = cnt_msg.view(E, B)

            x = scatter(message, node_out, dim=0, dim_size=x.size(0))
            min_cnt = scatter(cnt_msg, node_out, dim=0, dim_size=min_cnt.size(0), reduce="min")     

        return x, min_cnt
    
    def propagate(self, x, relation, edges_to_remove=None):
        device = x.device
        node_in = self.relation2adjacency[relation][0][1]
        node_out = self.relation2adjacency[relation][0][0]
        if device.type == "cuda":
            node_in = node_in.cuda(device)
            node_out = node_out.cuda(device)

        message = x[node_in]
        E, B, D = message.size()

        if edges_to_remove == None:
            x = scatter(message, node_out, dim=0, dim_size=x.size(0))
        else:
            # message: edge * batch * dim
            message = message.view(-1, D)
            bias = torch.arange(B)
            if device.type == "cuda":
                bias = bias.cuda(device)
            edges_to_remove = edges_to_remove * B + bias
            message[edges_to_remove] = 0
            message = message.view(E, B, D)
            x = scatter(message, node_out, dim=0, dim_size=x.size(0))

        return x