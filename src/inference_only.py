from run_rnnlogic import PredictorPlus, KnowledgeGraph, TrainerPredictor
from utils import load_config
import torch
from data.triples_dataset import TestDataset
import os

import ast

def load_readable_rules(rules_file, relations_dict_file):
    # 1. Load relation name -> id mapping
    name_to_id = {}
    with open(relations_dict_file, 'r', encoding='utf-8') as f:
        for line in f:
            rel_id, rel_name = line.strip().split('\t')
            name_to_id[rel_name] = int(rel_id)

    # 2. Read and parse rules
    rules = []
    with open(rules_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Parse the Python list string into a Python list of strings
            try:
                relation_names = ast.literal_eval(line.strip())
            except (ValueError, SyntaxError):
                continue  # skip invalid lines
            # Convert names back to IDs
            rule_ids = []
            for name in relation_names:
                rid = name_to_id.get(name)
                if rid is None:
                    raise KeyError(f"Relation name '{name}' not found in dictionary.")
                rule_ids.append(rid)
            rules.append(rule_ids)

    return rules

def main():
    config_path = "../config/full.yaml"
    cfgs = load_config(config_path)
    cfg = cfgs[0]
    result_dir = "./results/semmeddb/multitask_mmoe/ent_all_types/ablations/scheduled_sampling/use_scheduled_sampling" # TODO: 改這邊
    # result_dir = "./results/semmeddb/oris/ori1" # TODO: 改這邊

    # os.makedirs(f"{result_dir}/evals", exist_ok=True)
    save_dir = f"{result_dir}"
    # result_dir = "./results/semmeddb/oris/ori_1"

    device = torch.device("cuda:0")

    graph = KnowledgeGraph("../data/semmeddb", "type")
    test_set = TestDataset(graph, cfg.data.batch_size)
    predictor = PredictorPlus(graph, **cfg.predictorplus.model)

    rules = load_readable_rules(f"{result_dir}/rules_final.txt", "../data/semmeddb/relations.dict")

    predictor.test_set = test_set
    predictor.set_rules(rules)
    optim = torch.optim.Adam(predictor.parameters(), **cfg.predictorplus.optimizer)
    ckpt = torch.load(f"{result_dir}/predictor.pt", map_location=device, weights_only=True)

    # 加载模型参数
    predictor.load_state_dict(ckpt["model"])
    predictor.to(device)
    predictor.eval()


    solver_p = TrainerPredictor(predictor, test_set, test_set, test_set, optim, "xxx", gpus=[0])
    print("Evaluating on test set...")
    rel_metrics = solver_p.inference_evaluate('test') # 存 per_relation and overall
    rel_metrics.to_csv(f"{save_dir}/rel_metrics.csv", index=False)
    print(f"\n✅ Saved to {save_dir}")



if __name__ == "__main__":
    main()
