# ───────── data/rules_dataset.py ─────────
import torch
from torch.utils.data import Dataset
from functools import partial
import logging
# -------------------------------------------------------------
def load_relation_clusters(path):
    rel2cluster = {}
    with open(path) as fi:
        for ln in fi:
            r, c = ln.strip().split("\t")
            rel2cluster[int(r)] = int(c)
    return rel2cluster
# -------------------------------------------------------------

class RuleDataset(Dataset):
    """
    self.rules[i] = [rel_seq(+END), pad_idx, conf, <opt multihot(L,T)>]
    """
    def __init__(self, num_rel, inp, cluster_size, rel_cluster_file):
        # --------- 基本欄位 ----------
        self.num_relations = num_rel
        self.ending_idx    = num_rel
        self.padding_idx   = num_rel + 1

        self.cluster_size  = cluster_size
        self.rel2cluster   = load_relation_clusters(rel_cluster_file)

        # --------- 讀入 rule（一維） ----------
        if isinstance(inp, list):
            rules_flat = inp
        else:
            rules_flat = []
            with open(inp) as fi:
                for ln in fi:
                    *rels, conf = ln.split()
                    rules_flat.append([int(r) for r in rels] + [float(conf)*1000])

        self.rp_input = rules_flat                 # ← 這行必不可少

        # --------- 轉「三段式」 ----------
        self.rules = []
        for rf in rules_flat:
            conf = rf[-1]
            rels = rf[:-1] + [self.ending_idx]
            self.rules.append([rels, self.padding_idx, conf+1e-5])   # 先 3 欄

        # 預設 collate_fn（還沒有 multihot）
        self.type_size  = 1                        # 先設 1，之後 update 再改
        self.collate_fn = partial(self._collate_static, self.rel2cluster)

    # =============  更新 multi-hot  =============
    def update_grd_multihot(self, mh_list, type_size):
        """
        mh_list : List[ Tensor(L_i , type_size) ]  (順序 = self.rules)
        """
        assert len(mh_list) == len(self.rules)
        self.type_size = type_size                # ← 記住真正的 type 大小

        max_len = max(len(r[0]) - 1 for r in self.rules)  # rule_len(不含 END)

        for i, (mh, rec) in enumerate(zip(mh_list, self.rules)):
            # 1. 長度對齊
            if mh.size(0) < max_len:
                pad = torch.zeros(max_len - mh.size(0), type_size)
                mh  = torch.cat([mh, pad], 0)
            elif mh.size(0) > max_len:
                mh  = mh[:max_len]

            # 2. 寫回 rule item (變成 4 欄)
            if len(rec) == 3:
                rec.append(mh)
            else:
                rec[3] = mh

        # 3. 把 collate_fn 重新綁定（簽名沒變）
        self.collate_fn = partial(self._collate_static, self.rel2cluster)

    # ============= Dataset 介面 =============
    def __len__(self):
        return len(self.rules)

    def __getitem__(self, idx):
        return self.rules[idx]

    # ============= collate_fn =============
    @staticmethod
    def _collate_static(rel2cluster, batch):
        """
        batch 內 item 可能有 3 欄或 4 欄
        """
        # -------- 1. 拆包 --------
        has_mh = (len(batch[0]) == 4)
        if has_mh:
            rels, pad_idx, conf, mh = zip(*batch)
            multihots = list(mh)           # List[Tensor(L , T)]
            type_size = multihots[0].size(-1)
        else:
            rels, pad_idx, conf = zip(*batch)
            multihots = None
            type_size = 1                  # 占位

        # -------- 2. rule → seq / tgt --------
        inputs       = [r[:-1] for r in rels]    # 去掉 END
        main_target  = [r[1:]  for r in rels]
        max_len      = max(len(s) for s in inputs)
        B            = len(inputs)

        # -------- 3. relation-cluster target --------
        aux_rc = [[rel2cluster.get(r,-1) for r in tgt] for tgt in main_target]

        # -------- 4. multihot & padding --------
        sem_mh = torch.zeros(B, max_len, type_size)   # 全 0
        for i in range(B):
            pad_n = max_len - len(inputs[i])

            # multihot
            if multihots is not None:
                mh = multihots[i]                    # (L_i , T)
                sem_mh[i, :mh.size(0), :] = mh

            # rule 與其它目標 padding
            inputs[i]      += [pad_idx[i]] * pad_n
            main_target[i] += [pad_idx[i]] * pad_n
            aux_rc[i]      += [-1] * pad_n

        # right-shift 一格 → ent-type target
        aux_et_tgt = torch.zeros_like(sem_mh)
        aux_et_tgt[:, :-1, :] = sem_mh[:, 1:, :]

        # -------- 5. to tensor --------
        inputs      = torch.tensor(inputs,      dtype=torch.long)
        main_target = torch.tensor(main_target, dtype=torch.long)
        aux_rc      = torch.tensor(aux_rc,      dtype=torch.long)
        weight      = torch.tensor(conf,        dtype=torch.float)

        pad_id = pad_idx[0]
        mask   = inputs != pad_id

        logging.debug(f"sem_mh shape={sem_mh.shape}")  # (B,L,T)

        return {
            "sequence"               : inputs,
            "relation"               : inputs[:,0],
            "mask"                   : mask,
            "weight"                 : weight,
            "main_target"            : main_target,
            "aux_rel_cluster_target" : aux_rc,
            "aux_ent_type_multihot"  : sem_mh,      # (B,L,T)
            "aux_ent_type_target"    : aux_et_tgt,  # (B,L,T)
        }
