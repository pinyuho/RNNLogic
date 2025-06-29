import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA, TruncatedSVD


from collections import Counter, defaultdict
from data.graph import KnowledgeGraph

TYPE_SIZE = 136

class RelationClusterer:
    def __init__(self, dbname, cluster_size, workingdir, graph, entity_type_path, relation_dict_path):
        self.dbname = dbname
        self.graph = graph
        self.workingdir = workingdir
        self.ent2types = self.load_entity_type_dict(entity_type_path)
        self.id2rel = self.load_relation_dict(relation_dict_path)
        self.cluster_size = cluster_size

    def load_entity_type_dict(self, file_path):
        ent2types = {}
        # with open(file_path, "r", encoding="utf-8") as f:
        #     for line in f:
        #         eid, tid = line.strip().split('\t')
        #         ent2type[int(eid)] = int(tid)
        with gzip.open(file_path, "rb") as f:
            ent2types = pickle.load(f)   # 直接得到原 dict
        return ent2types

    def load_relation_dict(self, file_path):
        return {int(idx): rel for idx, rel in \
                (line.strip().split("\t") for line in open(file_path, "r", encoding="utf-8"))}

    # def encode_relation_features(self, mode="naive", svd=False, svd_dim=5):
    #     num_lexnames = max(self.ent2type.values()) + 1
    #     relation_to_pairs = defaultdict(list)

    #     for h, r, t in self.graph.train_facts:
    #         relation_to_pairs[r].append((h, t))

    #     if mode == "naive":
    #         relation_features = {}
    #         for r, pairs in relation_to_pairs.items():
    #             feats = [
    #                 F.one_hot(torch.tensor(self.ent2type.get(h, 0)), num_classes=num_lexnames).float() +
    #                 F.one_hot(torch.tensor(self.ent2type.get(t, 0)), num_classes=num_lexnames).float()
    #                 for h, t in pairs
    #             ]
    #             relation_features[r] = torch.stack(feats).mean(dim=0)

    #         rel_ids = list(relation_features.keys())
    #         X = torch.stack([relation_features[r] for r in rel_ids])
    #         return X, rel_ids
        
    #     elif mode == "matrix":
    #         relation_matrix = {r: torch.zeros(TYPE_SIZE, TYPE_SIZE) for r in relation_to_pairs.keys()}
    #         for r, pairs in relation_to_pairs.items():
    #             for h, t in pairs:
    #                 if r in [7, 12, 15, 16]: # bi-directional relations: compared_with, associated_with, interacts_with, coexists_with
    #                     relation_matrix[r][self.ent2type.get(h, -1)][self.ent2type.get(t, -1)] += 1
    #                     relation_matrix[r][self.ent2type.get(t, -1)][self.ent2type.get(h, -1)] += 1
    #                 else: # one-directional relations
    #                     relation_matrix[r][self.ent2type.get(h, -1)][self.ent2type.get(t, -1)] += 1

            
    #         rel_ids = list(relation_matrix.keys())
    #         X = torch.stack([relation_matrix[r] for r in rel_ids], dim=0)  # tensor_mat.shape == (num_rel, TYPE_SIZE, TYPE_SIZE)
    #         flat_X = X.view(len(rel_ids), -1)     # (num_rel, TYPE_SIZE**2)

    #         if svd == True:
    #             # svd = TruncatedSVD(n_components=512)
    #             # svd.fit(flat_X)
    #             # cum_var = svd.explained_variance_ratio_.cumsum()
    #             # # 找到 cum_var >= 0.90 的最小 dim，比如 80、120
    #             # # threshold = 0.90
    #             # dim_90 = np.searchsorted(cum_var, 0.90) + 1
    #             # print(f"需要 {dim_90} 維才能覆蓋 ≥90% 變異量") 

    #             svd_reducer = TruncatedSVD(n_components=svd_dim, random_state=42)
    #             flat_X_reduced  = svd_reducer.fit_transform(flat_X) 
    #             return flat_X_reduced, rel_ids

    #         return flat_X, rel_ids
    #     else:
    #         return None, None

    def encode_relation_features(self, mode="naive", svd=False, svd_dim=5):
        # 1. 計算 type vocab size  (max id + 1)
        num_type = max(t for ts in self.ent2types.values() for t in ts) + 1

        relation_to_pairs = defaultdict(list)
        for h, r, t in self.graph.train_facts:
            relation_to_pairs[r].append((h, t))

        # ==========  NAIVE  ==========
        if mode == "naive":
            relation_features = {}
            for r, pairs in relation_to_pairs.items():
                feats = []
                for h, t in pairs:
                    # ---- head multi-hot ----
                    h_vec = torch.zeros(num_type)
                    h_types = self.ent2types.get(h, [])
                    h_vec[h_types] = 1.0

                    # ---- tail multi-hot ----
                    t_vec = torch.zeros(num_type)
                    t_types = self.ent2types.get(t, [])
                    t_vec[t_types] = 1.0

                    feats.append(h_vec + t_vec)     # (B, T)
                relation_features[r] = torch.stack(feats).mean(dim=0)

            rel_ids = list(relation_features.keys())
            X = torch.stack([relation_features[r] for r in rel_ids])
            return X, rel_ids

        # ==========  MATRIX  ==========
        elif mode == "matrix":
            TYPE_SIZE = num_type
            relation_matrix = {
                r: torch.zeros(TYPE_SIZE, TYPE_SIZE)
                for r in relation_to_pairs
            }

            for r, pairs in relation_to_pairs.items():
                for h, t in pairs:
                    h_types = self.ent2types.get(h, [])
                    t_types = self.ent2types.get(t, [])

                    # 笛卡兒積：所有 (h_type, t_type) 組合都 +1
                    for ht in h_types:
                        for tt in t_types:
                            relation_matrix[r][ht][tt] += 1
                            # 雙向關係再加一次
                            if r in {7, 12, 15, 16}:          # compared_with 等
                                relation_matrix[r][tt][ht] += 1

            rel_ids = list(relation_matrix.keys())
            X = torch.stack([relation_matrix[r] for r in rel_ids])      # (R, T, T)
            flat_X = X.view(len(rel_ids), -1)

            if svd:
                flat_X = TruncatedSVD(
                    n_components=svd_dim, random_state=42
                ).fit_transform(flat_X)
            return flat_X, rel_ids

        else:
            return None, None


    def plot_elbow_curve(self, X, max_k):
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X

        sse = []
        sil_scores = []
        db_scores = []
        ch_scores = []

        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42).fit(X_np)
            labels = kmeans.labels_
            sse.append(kmeans.inertia_)
            sil_scores.append(silhouette_score(X_np, labels))
            db_scores.append(davies_bouldin_score(X_np, labels))
            ch_scores.append(calinski_harabasz_score(X_np, labels))

        # 畫四個指標的折線圖
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(range(2, max_k + 1), sse, marker='o')
        plt.title('SSE (Inertia)')
        plt.xlabel('Number of Clusters')
        plt.ylabel('SSE')

        plt.subplot(2, 2, 2)
        plt.plot(range(2, max_k + 1), sil_scores, marker='o')
        plt.title('Silhouette Score')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')

        plt.subplot(2, 2, 3)
        plt.plot(range(2, max_k + 1), db_scores, marker='o')
        plt.title('Davies-Bouldin Index')
        plt.xlabel('Number of Clusters')
        plt.ylabel('DB Index')

        plt.subplot(2, 2, 4)
        plt.plot(range(2, max_k + 1), ch_scores, marker='o')
        plt.title('Calinski-Harabasz Index')
        plt.xlabel('Number of Clusters')
        plt.ylabel('CH Index')

        plt.tight_layout()
        # save_path = os.path.join(self.dbname, f"cluster_{self.cluster_size}", "kmeans_scores_over_k.png")
        save_path = os.path.join(self.workingdir, "kmeans_scores_over_k.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"[Cluster Evaluation Plot] Saved to {save_path}")

    def run_kmeans(self, X, rel_ids):
        kmeans = KMeans(n_clusters=self.cluster_size, random_state=42).fit(X)
        labels = kmeans.labels_

        output_path = os.path.join(self.workingdir, "relation_cluster.dict")
        # output_path = os.path.join(self.dbname, f"test", "relation_cluster.dict")
        with open(output_path, "w") as f:
            for r, label in zip(rel_ids, labels):
                f.write(f"{r}\t{label}\n")
        print(f"[KMeans Result] Saved to {output_path}")

        return labels, {r: label for r, label in zip(rel_ids, labels)}

    def save_clustered_relations(self, rel2clus):
        clusters = defaultdict(list)
        for relid, clusterid in rel2clus.items():
            clusters[clusterid].append(self.id2rel[relid])

        # output_path = os.path.join(self.dbname, f"cluster_{self.cluster_size}", "clustered_rels.txt")
        output_path = os.path.join(self.workingdir, "clustered_rels.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            for cid in range(self.cluster_size):
                f.write(f"Cluster {cid}:\n")
                for rel in sorted(clusters[cid]):
                    f.write(f"  - {rel}\n")
                f.write("\n")
        print(f"[Cluster Dictionary] Saved to {output_path}")

    def evaluate_clustering(self, X, labels):
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        labels = np.array(labels)

        # --- Compute cluster centers
        unique_labels = np.unique(labels)
        centroids = np.array([
            X_np[labels == k].mean(axis=0) for k in unique_labels
        ])

        # --- Compute SSE
        sse = 0.0
        for k in unique_labels:
            cluster_points = X_np[labels == k]
            sse += np.sum((cluster_points - centroids[k]) ** 2)

        # --- Other metrics
        sil_score = silhouette_score(X_np, labels)
        db_score = davies_bouldin_score(X_np, labels)
        ch_score = calinski_harabasz_score(X_np, labels)

        summary = (
            f"\n[Cluster Validity (k = {self.cluster_size})]\n"
            f"- SSE (Inertia)            : {sse:.4f} (Lower is better)\n"
            f"- Silhouette Score         : {sil_score:.4f} (Higher is better, max=1.0)\n"
            f"- Davies-Bouldin Index     : {db_score:.4f} (Lower is better, min=0.0)\n"
            f"- Calinski-Harabasz Index  : {ch_score:.4f} (Higher is better)\n"
        )

        print(summary)

        save_path = os.path.join(self.workingdir, "cluster_validity.txt")
        with open(save_path, "w") as f:
            f.write(summary)
        print(f"[Cluster Validity] Saved to {save_path}")

        return {
            "sse": sse,
            "silhouette_score": sil_score,
            "davies_bouldin_index": db_score,
            "calinski_harabasz_index": ch_score
        }


    def visualize_pca_clusters(self, X, labels, method="pca", title="Cluster Visualization"):
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X

        reducer = PCA(n_components=3)
        X_reduced = reducer.fit_transform(X_np)

        unique_labels = np.unique(labels)
        num_labels = len(unique_labels)
        cmap = plt.get_cmap("tab10", num_labels)

        projections = [
            (0, 1, "PC1 vs PC2"),
            (1, 2, "PC2 vs PC3"),
            (0, 2, "PC1 vs PC3")
        ]

        for ix, iy, proj_title in projections:
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(X_reduced[:, ix], X_reduced[:, iy], c=labels, cmap=cmap, s=50, edgecolor="k")

            cbar = plt.colorbar(scatter, ticks=unique_labels)
            cbar.ax.set_yticklabels([str(l) for l in unique_labels])

            plt.title(f"{title} – {proj_title}")
            plt.xlabel(f"Component {ix+1}")
            plt.ylabel(f"Component {iy+1}")
            plt.grid(True)

            # save_path = os.path.join(self.dbname, f"cluster_{self.cluster_size}", f"embedding_pc{ix+1}_pc{iy+1}.png")
            save_path = os.path.join(self.workingdir, f"embedding_pc{ix+1}_pc{iy+1}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"[Embedding Visualization] Saved to {save_path}")


if __name__ == "__main__":

    dbname = "semmeddb"
    cluster_size = 3
    mode = "matrix"  # "naive", "matrix"
    svd_dim = 20

    for mode in ["naive", "matrix"]:
        for cluster_size in [3, 4, 5, 6, 7, 8]:
            if mode == "matrix":
                workingdir = f"semmeddb_alltypes_0629/{mode}/ori/cluster_{cluster_size}"
            else:
                workingdir = f"semmeddb_alltypes_0629/{mode}/cluster_{cluster_size}"
            # workingdir = f"{dbname}/{mode}/svd/dim_{svd_dim}"
            # workingdir = f"{dbname}/bidirectional/cluster_{cluster_size}"
            graph = KnowledgeGraph(f"../../data/{dbname}")

            clusterer = RelationClusterer(
                dbname=dbname,
                cluster_size=cluster_size,
                workingdir=workingdir,
                graph=graph,
                # entity_type_path=f"./entity_type.dict",
                entity_type_path="entid2typeids.pkl.gz",
                relation_dict_path=f"../../data/{dbname}/relations.dict"
            )

            X, rel_ids = clusterer.encode_relation_features(mode, svd=True, svd_dim=svd_dim)
            clusterer.plot_elbow_curve(X, max_k=12)

            labels, rel2clus = clusterer.run_kmeans(X, rel_ids)
            clusterer.save_clustered_relations(rel2clus)
            clusterer.evaluate_clustering(X, labels)

            clusterer.visualize_pca_clusters(X, labels, method="pca", title="KMeans Clustering")