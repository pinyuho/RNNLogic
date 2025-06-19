import pandas as pd
from collections import defaultdict
from tqdm import tqdm

def load_relation_dict(relation_path):
    id2rel, rel2id = {}, {}
    with open(relation_path, "r", encoding="utf-8") as f:
        for line in f:
            rid, name = line.rstrip().split("\t")
            rid = int(rid)
            id2rel[rid] = name
            rel2id[name] = rid
    return id2rel, rel2id

def load_rules_df(rule_path, dict_path="../data/semmeddb/relations.dict"):
    """
    讀 mined_rules.txt  
    格式:  h_id  b1  b2  …  bk  conf   (空白分隔)
    回傳欄位: r_head(str), body(list[str]), conf(float)
    """
    id2rel, _ = load_relation_dict(dict_path)

    rows = []
    with open(rule_path, "r") as fi:
        for line in fi:
            parts = line.rstrip().split()         # ★ ← 空白分隔
            if len(parts) < 3:                    # 至少 head + 1 body + conf
                continue

            head_id  = int(parts[0])
            body_ids = list(map(int, parts[1:-1]))
            conf     = float(parts[-1])

            # id → 名稱；若有不存在的 id 就跳過
            try:
                head_rel  = id2rel[head_id]
                body_rels = [id2rel[rid] for rid in body_ids]
            except KeyError:
                continue

            rows.append({"r_head": head_rel,
                         "body":   body_rels,
                         "conf":   conf})

    return pd.DataFrame(rows)

def load_triples(triple_path):
    df = pd.read_csv(triple_path, sep="\t", header=None, names=["h", "r", "t"])
    # 建兩個索引：  head→(r→tails)   &   mid→(r→tails)   方便多 hop
    head2rt = defaultdict(lambda: defaultdict(set))
    for h, r, t in df.itertuples(index=False):
        head2rt[h][r].add(t)
    return head2rt, set(map(tuple, df.values))  

def apply_rule(head2rt, h, body_rels):
    """從 head h 沿 body_rels 走，多 hop 聚合所有可達 tail"""
    cur_nodes = {h}
    for rel in body_rels:
        nxt = set()
        for node in cur_nodes:
            nxt |= head2rt[node].get(rel, set())
        if not nxt:       # 走不下去
            return set()
        cur_nodes = nxt
    return cur_nodes  

def rule_accuracy(rule_row, head2rt, gt_set):
    r_head   = rule_row.r_head
    body_rels= rule_row.body
    TP = FP = FN = 0

    for h in list(head2rt.keys()): 
        # ① 規則推到哪些 tail
        tails_pred = apply_rule(head2rt, h, body_rels)

        # ② 真實圖裡該 (h, r_head, ?) 的 tail
        tails_true = head2rt[h].get(r_head, set())

        # ③ 統計
        for t in tails_pred:
            if (h, r_head, t) in gt_set:
                TP += 1
            else:
                FP += 1
        FN += len(tails_true - tails_pred)   # 真實有但規則沒推到

    denom = TP + FP + FN
    # print(f"{r_head} {body_rels} | TP: {TP}, FP: {FP}, FN: {FN}, denom: {denom}")
    return 0.0 if denom == 0 else TP / denom

def row_to_line(row):
    _, rel2id = load_relation_dict("../data/semmeddb/relations.dict")
    head_id  = rel2id[row["r_head"]]
    body_ids = [str(rel2id[r]) for r in row["body"]]    # list[str]
    conf    = f"{row['conf']}"                # 或 row['conf']
    acc    = f"{row['accuracy']:.10f}"                # row['acc']
    return " ".join([str(head_id), *body_ids, conf, acc])

def main(rule_path, triple_path, save_path, top_k=100):
    head2rt, gt_set = load_triples(triple_path)
    rules_df        = load_rules_df(rule_path)

    accs = []
    for rule in tqdm(rules_df.itertuples(index=False), total=len(rules_df)):
        # print(f"Processing rule: {rule.r_head} {rule.body}")
        accs.append(rule_accuracy(rule, head2rt, gt_set))
    rules_df["accuracy"] = accs

    top_df = rules_df.sort_values("accuracy", ascending=False).head(top_k)

    with open(save_path, "w") as fo:
        for _, row in top_df.iterrows():
            fo.write(row_to_line(row) + "\n")

    print(f"Saved {len(top_df)} rules → {save_path}")

if __name__ == "__main__":
    main(
        rule_path = "../data/semmeddb/mined_rules.txt",
        triple_path = "../data/semmeddb/train_filtered.txt",
        save_path = "../data/semmeddb/weitsu_filter_rules_accu.txt",
        top_k = 50
    )

