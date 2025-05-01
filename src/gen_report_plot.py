import matplotlib.pyplot as plt

# 模擬之前的資料
models = ["RNNLogic", "multitask_fixed", "multitask_warmup", "multitask_schedule", "multitask_adaptive", "multitask_adaptive_topk"]
sizes = ["small", "mid", "full"]
data = {
    "RNNLogic": [0.468, 0.456, 0.481],
    "multitask_fixed": [0.443, 0.449, None],
    "multitask_warmup": [0.465, 0.446, None],
    "multitask_schedule": [0.462, 0.452, None],
    "multitask_adaptive": [0.473, 0.443, 0.478],
    "multitask_adaptive_topk": [0.470, 0.448, 0.476],
}

# 繪製折線圖
plt.figure(figsize=(10, 6))

for model in models:
    y = data[model]
    plt.plot(sizes, y, marker='o', label=model)

plt.title("MRR across Model Sizes")
plt.xlabel("Model Size")
plt.ylabel("MRR")
plt.ylim(0.44, 0.49)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig("mrr_report")