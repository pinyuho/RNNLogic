import torch
from torch.utils.data import Dataset
import random

class TrainDataset(Dataset):
    def __init__(self, graph, batch_size):
        self.graph = graph
        self.batch_size = batch_size
        self.triple2count = graph.triple2count
        self.default_cnt = graph.default_cnt

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

        triple_weight = torch.FloatTensor([
            self.triple2count.get((h, r, t), self.default_cnt) for (h, r, t) in data
        ])   # shape (B,)
        triple_weight = triple_weight.unsqueeze(1)    

        return all_h, all_r, all_t, target, edges_to_remove, triple_weight

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
