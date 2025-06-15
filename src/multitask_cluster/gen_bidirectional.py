target_ids = {7, 12, 15, 16}
output_path = "semmeddb/bidirectional/relation_cluster.dict"

with open(output_path, "w") as f:
    for rid in range(20):
        cluster_id = 1 if rid in target_ids else 0
        f.write(f"{rid}\t{cluster_id}\n")

print(f"[Done] Saved to {output_path}")
