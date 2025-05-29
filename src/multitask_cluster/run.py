import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA


from collections import Counter, defaultdict
from data import KnowledgeGraph

class RelationClusterer:
    def __init__(self, dbname, graph, cluster_size, entity_type_path, relation_dict_path):
        self.dbname = dbname
        self.graph = graph
        self.entityid_to_typeid = self.load_entity_type_dict(entity_type_path)
        self.id2rel = self.load_relation_dict(relation_dict_path)
        self.cluster_size = cluster_size

    def set_cluster_num(self, num_clusters):
        self.num_clusters = num_clusters

    def load_entity_type_dict(self, file_path):
        entityid_to_typeid = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                eid, tid = line.strip().split('\t')
                entityid_to_typeid[int(eid)] = int(tid)
        return entityid_to_typeid

    def load_relation_dict(self, file_path):
        return {int(idx): rel for idx, rel in \
                (line.strip().split("\t") for line in open(file_path, "r", encoding="utf-8"))}

    def encode_relation_features(self):
        num_lexnames = max(self.entityid_to_typeid.values()) + 1
        relation_to_pairs = defaultdict(list)

        for h, r, t in self.graph.train_facts:
            relation_to_pairs[r].append((h, t))

        relation_features = {}
        for r, pairs in relation_to_pairs.items():
            feats = [
                F.one_hot(torch.tensor(self.entityid_to_typeid.get(h, 0)), num_classes=num_lexnames).float() +
                F.one_hot(torch.tensor(self.entityid_to_typeid.get(t, 0)), num_classes=num_lexnames).float()
                for h, t in pairs
            ]
            relation_features[r] = torch.stack(feats).mean(dim=0)

        rel_ids = list(relation_features.keys())
        X = torch.stack([relation_features[r] for r in rel_ids])
        return X, rel_ids

    # def plot_elbow_curve(self, X, max_k):
    #     X_np = X.numpy() if isinstance(X, torch.Tensor) else X
    #     sse = [KMeans(n_clusters=k, random_state=42).fit(X_np).inertia_ for k in range(1, max_k + 1)]

    #     plt.plot(range(1, max_k + 1), sse, marker='o')
    #     plt.xlabel("Number of Clusters (K)")
    #     plt.ylabel("SSE (Inertia)")
    #     plt.title("Elbow Method for Optimal K")
    #     plt.grid(True)
    #     save_path = os.path.join(self.dbname, "kmeans_elbow.png")
    #     plt.savefig(save_path)
    #     print(f"[Elbow Plot] Saved to {save_path}")
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
        save_path = os.path.join(self.dbname, f"cluster_{self.cluster_size}", "kmeans_scores_over_k.png")
        plt.savefig(save_path)
        print(f"[Cluster Evaluation Plot] Saved to {save_path}")


    def run_kmeans(self, X, rel_ids):
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42).fit(X.numpy())
        labels = kmeans.labels_

        output_path = os.path.join(self.dbname, f"cluster_{self.cluster_size}", "relation_cluster.dict")
        with open(output_path, "w") as f:
            for r, label in zip(rel_ids, labels):
                f.write(f"{r}\t{label}\n")
        print(f"[KMeans Result] Saved to {output_path}")

        return labels, {r: label for r, label in zip(rel_ids, labels)}

    def save_clustered_relations(self, rel2clus):
        clusters = defaultdict(list)
        for relid, clusterid in rel2clus.items():
            clusters[clusterid].append(self.id2rel[relid])

        output_path = os.path.join(self.dbname, f"cluster_{self.cluster_size}", "clustered_rels.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            for cid in range(self.num_clusters):
                f.write(f"Cluster {cid}:\n")
                for rel in sorted(clusters[cid]):
                    f.write(f"  - {rel}\n")
                f.write("\n")
        print(f"[Cluster Dictionary] Saved to {output_path}")

    def evaluate_clustering(self, X, labels):
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        sil_score = silhouette_score(X_np, labels)
        db_score = davies_bouldin_score(X_np, labels)
        ch_score = calinski_harabasz_score(X_np, labels)

        summary = (
            f"\n[Cluster Validity (k = {self.num_clusters})]\n"
            f"- Silhouette Score        : {sil_score:.4f} (Higher is better, max=1.0)\n"
            f"- Davies-Bouldin Index     : {db_score:.4f} (Lower is better, min=0.0)\n"
            f"- Calinski-Harabasz Index  : {ch_score:.4f} (Higher is better)\n"
        )

        print(summary)

        save_path = os.path.join(self.dbname, f"cluster_{self.cluster_size}", "cluster_validity.txt")
        with open(save_path, "w") as f:
            f.write(summary)
        print(f"[Cluster Validity] Saved to {save_path}")
        return {"silhouette_score": sil_score, "davies_bouldin_index": db_score, "calinski_harabasz_index": ch_score}

    def visualize_clusters(self, X, labels, method="pca", title="Cluster Visualization"):
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

            save_path = os.path.join(self.dbname, f"cluster_{self.cluster_size}", f"embedding_pc{ix+1}_pc{iy+1}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"[Embedding Visualization] Saved to {save_path}")


if __name__ == "__main__":

    dbname = "semmeddb"
    cluster_size = 3
    graph = KnowledgeGraph(f"../../data/{dbname}")

    clusterer = RelationClusterer(
        dbname=dbname,
        graph=graph,
        cluster_size=cluster_size,
        entity_type_path=f"{dbname}/entity_type.dict",
        relation_dict_path=f"../../data/{dbname}/relations.dict"
    )

    X, rel_ids = clusterer.encode_relation_features()
    clusterer.plot_elbow_curve(X, max_k=12)

    clusterer.set_cluster_num(cluster_size)  # Replace this based on elbow curve observation
    labels, rel2clus = clusterer.run_kmeans(X, rel_ids)
    clusterer.save_clustered_relations(rel2clus)
    clusterer.evaluate_clustering(X, labels)

    clusterer.visualize_clusters(X, labels, method="pca", title="KMeans Clustering")
