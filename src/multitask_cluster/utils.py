from collections import Counter, defaultdict

# import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from sklearn.cluster import KMeans
import torch.nn.functional as F
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data import KnowledgeGraph
import torch


def get_first_lexname(offset_id):
    offset_id = int(offset_id)
    for pos in ['n', 'v', 'a', 'r', 's']:
        try:
            synset = wn.synset_from_pos_and_offset(pos, offset_id)
            return synset.lexname()
        except:
            continue
    return None

from collections import Counter

def generate_lexname_dict(entity_dict_file, output_type_file="types.dict", output_entity_lex_file="entity_lexname.dict"):
    lexname2id = {}
    entity_lex_list = []
    lexname_id_counter = Counter()

    with open(entity_dict_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        eid, offset_str = line.strip().split('\t')
        lexname = get_first_lexname(offset_str)

        if lexname is None:
            lexname_id = -1  # unknown category
        else:
            if lexname not in lexname2id:
                lexname2id[lexname] = len(lexname2id)
            lexname_id = lexname2id[lexname]

        entity_lex_list.append((eid, lexname_id))
        lexname_id_counter[lexname_id] += 1

    # save lexname.dict
    with open(output_type_file, "w") as f:
        for lexname, lid in sorted(lexname2id.items(), key=lambda x: x[1]):
            f.write(f"{lid}\t{lexname}\n")

    # save entities_lexname.dict
    with open(output_entity_lex_file, "w") as f:
        for eid, lid in entity_lex_list:
            f.write(f"{eid}\t{lid}\n")

    # 印出類別分佈統計
    print("Lexname ID 分佈：")
    for type_id, count in sorted(lexname_id_counter.items(), key=lambda x: x[1], reverse=True):
        print(f"Lexname ID {type_id}: {count} 個")

    return lexname_id_counter

def plot_elbow_curve(X, max_k, title, save_path):
    """
    繪製 Elbow 曲線來判斷最佳的 k 值（聚類數量）
    :param X: 要聚類的特徵矩陣 (torch.Tensor or numpy.ndarray)
    :param max_k: 嘗試的最大 k 值
    :param title: 圖片標題
    """
    if isinstance(X, torch.Tensor):
        X = X.numpy()

    sse = []
    k_range = range(1, max_k + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
        sse.append(kmeans.inertia_)

    plt.figure()
    plt.plot(k_range, sse, marker='o')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("SSE (Inertia)")
    plt.title(title)
    plt.grid(True)
    # plt.show()
    plt.savefig(save_path)  # 儲存圖檔
    print(f"[Elbow Plot] saved to {save_path}")

def generate_relation_cluster_dict(triples, entityid_to_typeid, num_clusters, dbname):
    num_lexnames = max(entityid_to_typeid.values()) + 1

    # 1. 收集每個 relation 對應的 (h, t)
    relation_to_pairs = defaultdict(list)
    for h, r, t in triples:
        relation_to_pairs[r].append((h, t))

    # 2. 建立 feature（h_lex + t_lex 的 one-hot sum）
    relation_features = {}
    for r, pairs in relation_to_pairs.items():
        feats = []
        for h, t in pairs:
            h_lex = entityid_to_typeid.get(h, 0)
            t_lex = entityid_to_typeid.get(t, 0)
            h_onehot = F.one_hot(torch.tensor(h_lex), num_classes=num_lexnames).float()
            t_onehot = F.one_hot(torch.tensor(t_lex), num_classes=num_lexnames).float()
            feats.append(h_onehot + t_onehot)
        relation_features[r] = torch.stack(feats).mean(dim=0)

    # 3. 聚類
    rel_ids = list(relation_features.keys())
    X = torch.stack([relation_features[r] for r in rel_ids])
    plot_elbow_curve(X, max_k=12, title="Elbow Method for Optimal k", save_path=f"{dbname}/kmeans_elbow.png")

    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(X.numpy())
    cluster_labels = kmeans.labels_

    # 4. 輸出對應表
    output_path = f"{dbname}/relation_cluster.dict"
    with open(output_path, "w") as f:
        rel2clus = {r: cid for r, cid in zip(rel_ids, cluster_labels)}
        for r in sorted(rel2clus):
            f.write(f"{r}\t{rel2clus[r]}\n")

    print(f"Saved to {output_path} ✅")
    return rel2clus

def load_entity_type_dict(file_path):
    entityid_to_typeid = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ent_id_str, type_id_str = line.split('\t')
            entityid_to_typeid[int(ent_id_str)] = int(type_id_str)
    return entityid_to_typeid

def get_clustered_rels_report(relid2clusterid, id2rel, num_clusters, dbname):
    res = dict()
    for i in range(num_clusters):
        res[i] = []
    for relid, clusterid in relid2clusterid.items():
        res[clusterid].append(id2rel[relid])

    output_path = f"{dbname}/clustered_rels.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for cluster_id in sorted(res):
            f.write(f"Cluster {cluster_id}:\n")
            for relation in sorted(res[cluster_id]):
                f.write(f"  - {relation}\n")
            f.write("\n")  # 每個 cluster 間空一行
    print(f"[✓] Cluster dictionary saved to {output_path}")
    

if __name__ == "__main__":
    # generate_lexname_dict("../../data/wn18rr/entities.dict")

    dbname = "semmed"

    graph = KnowledgeGraph(f"../../data/{dbname}")
    # print(graph.train_facts)
    
    entityid_to_typeid = load_entity_type_dict(f"{dbname}/entity_type.dict")

    id2rel = {}
    with open(f"../../data/{dbname}/relations.dict", "r", encoding="utf-8") as f:
        for line in f:
            idx, rel = line.strip().split("\t")
            id2rel[int(idx)] = rel

    # Step 1
    num_clusters = 4
    rel2clus = generate_relation_cluster_dict(graph.train_facts, entityid_to_typeid, num_clusters, dbname)

    # Step 2
    get_clustered_rels_report(rel2clus, id2rel, num_clusters, dbname)



