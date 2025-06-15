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

    best_mrr = -1
    match = re.search(r'Final Test MRR:\s*([0-9.]+)', log)
    if match:
        best_mrr = float(match.group(1))
        print("Extracted MRR:", best_mrr)

    round_digits = 4
    for match in pattern.finditer(log):
        iteration, valid_mrr, test_hit1, test_hit3, test_hit10, test_mr, test_mrr = match.groups()
        test_mrr = float(test_mrr)

        # 檢查是否為 best_mrr
        if abs(test_mrr - best_mrr) < 1e-6:  # 用誤差容忍避免浮點數誤差
            best_row = {
                "Model": model_name,
                "MR": round(float(test_mr), round_digits),
                "MRR": round(test_mrr, round_digits),
                "Hit@1": round(float(test_hit1), round_digits),
                "Hit@3": round(float(test_hit3), round_digits),
                "Hit@10": round(float(test_hit10), round_digits),
            }
            break  # 如果只取第一個匹配到的就好，這行可保留

    return best_row


# 範例用法
cluster_size = 5
# rootpath = f"../results/semmeddb/multitask/cluster_{cluster_size}"

# for size in ["sm", "md", "full"]:
for size in ["full"]:
    all_results = []
    # for model in ["ori", "fixed", "warmup", "schedule", "adaptive"]:
    for model in ["fixed", "warmup", "schedule", "adaptive"]:
        # log_path = f"{rootpath}/{size}/{model}/run.log"
        log_path = f"../results/semmeddb/sm_test/random_init/naive_cluster_{cluster_size}/{model}/run.log"
        best_row = extract_best_mrr_from_log(log_path)
        if best_row is not None:
            all_results.append(best_row)

    df_all = pd.DataFrame(all_results)

    filename = f"naive_cluster_{cluster_size}.csv"
    df_all.to_csv(filename, index=False)
    print(f"✅ {size} best results saved to {filename}")