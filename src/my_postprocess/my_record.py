import re
import pandas as pd
# 這段程式碼用來從訓練日誌中提取最佳的 MRR (Mean Reciprocal Rank) 和其他指標

def extract_best_mrr_from_log(log_path):
    with open(log_path, 'r') as f:
        log = f.read()

    model_name = log_path.split('/')[-2].replace('.log', '')

    pattern = re.compile(
        r"Iteration: (\d+)/\d+\s+.*?"
        r"Predictor: Evaluating on valid\s+.*?"
        r"Hit1\s+:\s+[\d.]+.*?"
        r"Hit3\s+:\s+[\d.]+.*?"
        r"Hit10:\s+[\d.]+.*?"
        r"MR\s+:\s+[\d.]+.*?"
        r"MRR\s+:\s+([\d.]+).*?"
        r"Predictor: Evaluating on test\s+.*?"
        r"Hit1\s+:\s+([\d.]+).*?"
        r"Hit3\s+:\s+([\d.]+).*?"
        r"Hit10:\s+([\d.]+).*?"
        r"MR\s+:\s+([\d.]+).*?"
        r"MRR\s+:\s+([\d.]+)",
        re.DOTALL
    )

    best_row = None
    best_valid_mrr = -1

    for match in pattern.finditer(log):
        iteration, valid_mrr, test_hit1, test_hit3, test_hit10, test_mr, test_mrr = match.groups()
        valid_mrr = float(valid_mrr)

        if valid_mrr > best_valid_mrr:
            best_valid_mrr = valid_mrr
            best_row = {
                "Model": model_name,
                "MR": round(float(test_mr), 3),
                "MRR": round(float(test_mrr), 3),
                "Hit@1": round(float(test_hit1), 3),
                "Hit@3": round(float(test_hit3), 3),
                "Hit@10": round(float(test_hit10), 3),
            }

    return best_row


# 範例用法
rootpath = "../results/semmeddb/multitask/cluster_4"
# for size in ["sm", "md", "full"]:
for size in ["full"]:
    all_results = []
    # for model in ["ori", "fixed", "warmup", "schedule", "adaptive"]:
    for model in ["fixed", "warmup", "schedule", "adaptive"]:
        log_path = f"{rootpath}/{size}/{model}/run.log"
        best_row = extract_best_mrr_from_log(log_path)
        if best_row is not None:
            all_results.append(best_row)

    df_all = pd.DataFrame(all_results)
    df_all.to_csv(f"all_best_results_{size}_cluster_4.csv", index=False)
    print(f"✅ {size} best results saved to all_best_results.csv")