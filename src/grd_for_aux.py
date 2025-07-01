import torch
import gzip
import json

import logging

from collections import Counter
from typing import Union, List, Tuple, Dict

def collect_multi_rule_ent_types(records, rules, dedup=True):
    """
    records : list of dict  (每行 json 轉成的 dict)
    rules   : list of rule ；每條 rule = [rule_head, r1, r2, ...]
    返回    : rule2ents  = { rule_tuple : [[heads], [tails_r1], [tails_r2], ...] } 
    
    e.g. {(5, 1, 4, 9): [[2], [10, 28], [33, 55], [100, 30]], (7, 2, 2): [[3], [8], [42]]}
    
    備註：rule_tuple = tuple(rule)  方便當 dict key
    """
    # ---------- 先把所有 rule body 做成 set 方便比對 ----------
    rule_set = {tuple(r[1:]) : tuple(r) for r in rules}   # body_tuple → full_rule_tuple

    # 暫存：{ full_rule_tuple : round_idx → [entity,…] }
    cache = {}

    for rec in records:
        body = tuple(rec.get("rule", []))     # record 只存 rule_body
        full = rule_set.get(body)
        if not full:                          # 不在目標 rule 列表
            continue
        if rec.get("label") != 1:             # 只要正確路徑
            continue
        path = rec.get("path")
        if not path or path[0][0] != rec["head"]:
            continue

        # 取用 / 初始化此 rule 的 round dict
        rdict = cache.setdefault(full, {})
        
        # round 0：head entity
        rdict.setdefault(0, []).append(rec["head"])

        # round 1..n：各 hop tail
        for rnd, (_, _, tailE) in enumerate(path, 1):
            if rnd > len(body):       # path 比 rule 長就不再收
                break
            rdict.setdefault(rnd, []).append(tailE)

    # ---------- 整理成 list[list] ----------
    rule2ents = {}
    for rule_full, rdict in cache.items():
        max_round = len(rule_full) - 1
        ent_rounds = []
        for rnd in range(max_round + 1):
            ents = rdict.get(rnd, [])
            if dedup:
                ents = list(set(ents))
            ent_rounds.append(ents)
        rule2ents[rule_full] = ent_rounds

    return rule2ents

def ent_rounds_to_multihot(ent_rounds, ent2types, num_type, dedup=True):
    """
    ent_rounds : list[list[int]]   每 round 的 entity id
    ent2types  : dict  entity -> type 或 [type1,type2,…]
    num_type   : type vocab size
    返回       : (type_rounds, multihot_rounds)
      type_rounds     = [[t1,t2,…], …]          # round 對應的 type id
      multihot_rounds = [ Tensor(num_type), … ] # 每 round 的 multi-hot
    """
    type_rounds = []
    multihot_rounds = []

    for ents in ent_rounds:
        t_list = []
        for e in ents:
            t = ent2types.get(e)
            if t is None:
                continue
            if isinstance(t, (list, tuple, set)):
                t_list.extend(t)
            else:
                t_list.append(t)

        if dedup:
            t_list = list(set(t_list))

        type_rounds.append(t_list)

        vec = torch.zeros(num_type)
        if t_list:
            vec[t_list] = 1.0         # 把那些 type 位置設成 1
        multihot_rounds.append(vec)

    return type_rounds, multihot_rounds

# def rules_to_encoding(rule2ents, rules, ent2type, num_type, dedup=True, soft_label=False): # rule2ents 是前面 collect_multi_rule_ent_types 的結果
def rules_to_encoding(rule2ents, rules, ent2types, num_type, dedup=True, is_soft_label=False): # rule2ents 是前面 collect_multi_rule_ent_types 的結果
    # multihot or 機率分佈
    """
    依 rules 順序回傳 multihot_rounds_list
      multihot_rounds_list[i] = 該 rule 每 round 的 multi-hot Tensor
                                形狀：(round+1, num_type)
    若某條 rule 在 rule2ents 中找不到 → 回傳全零矩陣
    """
    all_encodings = []

    for rule in rules:
        key = tuple(rule)
        ent_rounds = rule2ents.get(key)

        if ent_rounds is None:                     # 找不到 → 全零
            zero = torch.zeros(len(rule), num_type)
            all_encodings.append(zero)
            continue

        if not is_soft_label: # multihot
            # _, parts = ent_rounds_to_multihot(ent_rounds, ent2type, num_type, dedup=dedup)
            _, parts = ent_rounds_to_multihot(ent_rounds, ent2types, num_type, dedup=dedup)
        else:
            # _, parts = ent_rounds_to_type_dist(ent_rounds, ent2type, num_type, dedup=dedup)
            _, parts = ent_rounds_to_type_dist(ent_rounds, ent2types, num_type, dedup=dedup)

        parts = torch.stack(parts)           # 堆成 (round, num_type)
        all_encodings.append(parts)
        
    return all_encodings

# def grd2encoding(rules, dump_path, ent2type, type_size, soft_label=False):
def grd2encoding(rules, dump_path, ent2types, type_size, is_soft_label=False):
    # logging.info(f"type size: {type_size}")
    records = []                              # 每一輪 EM 都要讀新的
    with gzip.open(dump_path, "rt") as fp:
        for line in fp:
            records.append(json.loads(line))  # 每筆 append
            
    # logging.info(f"readed records: {records}")
    rule2ents = collect_multi_rule_ent_types(records, rules)
    # rule2multi = rules_to_encoding(rule2ents, rules, ent2type, type_size, soft_label)
    rule2multi = rules_to_encoding(rule2ents, rules, ent2types, type_size, is_soft_label)

    return rule2multi

# --------------- 假資料 -----------------
# records = [
#     # --- Rule 1 : R5 <- R1,R4,R9  (rule_body=[1,4,9]) ---
#     {"rule":[1,4,9], "head":2,
#      "path":[(2,1,10),(10,4,55),(55,9,100)],
#      "label":1},
#     {"rule":[1,4,9], "head":2,
#      "path":[(2,1,28),(28,4,33),(33,9,30)],
#      "label":1},
#     {"rule":[1,4,9], "head":2,               # label=0 → 應被過濾
#      "path":[(2,1,10),(10,4,55),(55,9,888)],
#      "label":0},

#     # --- Rule 2 : R7 <- R2,R2  (rule_body=[2,2]) ---
#     {"rule":[2,2],   "head":3,
#      "path":[(3,2,8),(8,2,42)],
#      "label":1},
#     {"rule":[2,2],   "head":3,               # label=0 → 應被過濾
#      "path":[(3,2,8),(8,2,13)],
#      "label":0},
# ]

# rules = [
#     [5, 1, 4, 9],        # R5 ← R1,R4,R9
#     [7, 2, 2],           # R7 ← R2,R2
# ]

# # entity→type 映射（隨便編）
# ent2type = {
#     2:5, 10:6, 28:3, 55:10, 33:5, 100:3, 30:8,
#     3:7, 8:1, 42:9
# }
# num_type = 12   # 假設 type id 只會落在 0~11

# # --------------- 呼叫工具函式 -----------------
# rule2ents = collect_multi_rule_ent_types(records, rules)
# rule2multi = rules_to_encoding(rule2ents, rules, ent2type, num_type)

# # --------------- 印結果 -----------------
# for r, multihot in zip(rules, rule2multi):
#     print("rule:", r)
#     print("multihot shape:", tuple(multihot.shape))  # (round+1, num_type)
#     for idx, vec in enumerate(multihot):
#         ones = torch.nonzero(vec).squeeze(-1).tolist()
#         print(f"  round{idx}: 1s at type idx -> {ones}")
#         print("multihot:", multihot)


# output
# rule: [5, 1, 4, 9]
# multihot shape: (4, 12)
#   round0: 1s at type idx -> [5]
#   round1: 1s at type idx -> [3, 6]
#   round2: 1s at type idx -> [5, 10]
#   round3: 1s at type idx -> [3, 8]

# rule: [7, 2, 2]
# multihot shape: (3, 12)
#   round0: 1s at type idx -> [7]
#   round1: 1s at type idx -> [1]
#   round2: 1s at type idx -> [9]


def ent_rounds_to_type_dist(
    ent_rounds: List[List[int]],
    ent2types: Dict[int, List[int]],
    num_type: int,
    dedup: bool = False
) -> Tuple[List[List[int]], List[torch.Tensor]]:
    """
    ent_rounds : [[e11, e12, …], [e21, …], …]  # 每 round 的 entity id
    ent2types  : {entity: type 或 [type1,type2,…]}
    num_type   : type vocab size
    dedup      : 若 True 先去重再計次數（→ 等權分布）

    回傳:
        type_rounds  : [[t1, t2, …], …]               # 每 round 的 type id
        dist_rounds  : [Tensor(num_type), …]          # 出現次數比例分布
    """
    type_rounds, dist_rounds = [], []

    for ents in ent_rounds:
        # ---- 取出所有 type ----
        t_list = []
        for e in ents:
            t = ent2types.get(e)
            if t is None:
                continue
            t_list.extend(t) if isinstance(t, (list, tuple, set)) else t_list.append(t)

        if dedup:
            t_list = list(set(t_list))               # 先去重再算頻率 → 各 type 只算一次

        type_rounds.append(t_list)

        # ---- 轉成比例向量 ----
        vec = torch.zeros(num_type)
        if t_list:                                   # 有任何 type 才計算
            cnt = Counter(t_list)                    # {type_id: 次數}
            total = len(t_list)                      # denominator
            for t_id, c in cnt.items():
                vec[t_id] = c / total               # 機率 = 次數 / 總數
        dist_rounds.append(vec)

    return type_rounds, dist_rounds
