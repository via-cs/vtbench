import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cdist

# === CONFIGURATION ===
data_root = "/Users/akkumy/Downloads/test_vtbench/vtbench/data"
datasets = ["Beef", "BeetleFly"]
output_dir = "results/raw_overlay_no_tslearn"
os.makedirs(output_dir, exist_ok=True)

# === 1. Simple .ts file loader (UCR format) ===
def load_tsfile_to_dataframe(path):
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    data = []
    labels = []
    for line in lines:
        if "@" in line or "#" in line:
            continue
        parts = line.split(",")
        labels.append(parts[-1])
        data.append(np.array([float(x) for x in parts[:-1]]))
    return np.array(data, dtype=object), np.array(labels)

# === 2. Basic DTW function ===
def dtw_distance(s1, s2):
    n, m = len(s1), len(s2)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m]

# === 3. Approximate DTW barycenter averaging ===
def dtw_barycenter(samples, max_iter=5):
    mean = np.mean(np.vstack([s[:min(map(len, samples))] for s in samples]), axis=0)
    for _ in range(max_iter):
        aligned = []
        for s in samples:
            path = dtw_path(mean, s)
            aligned_s = np.array([s[j] for _, j in path])
            aligned.append(np.interp(np.linspace(0, len(aligned_s)-1, len(mean)),
                                     np.arange(len(aligned_s)), aligned_s))
        mean = np.mean(aligned, axis=0)
    return mean

# Helper for DTW alignment
def dtw_path(s1, s2):
    n, m = len(s1), len(s2)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    # Traceback
    i, j = n, m
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        choices = [D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]]
        step = np.argmin(choices)
        if step == 0: i -= 1
        elif step == 1: j -= 1
        else:
            i -= 1
            j -= 1
    return path[::-1]

# === MAIN LOOP ===
results = []

for dataset in datasets:
    print(f"\n=== Processing {dataset} ===")
    train_path = os.path.join(data_root, dataset, f"{dataset}_TRAIN.ts")
    test_path = os.path.join(data_root, dataset, f"{dataset}_TEST.ts")

    X_train, y_train = load_tsfile_to_dataframe(train_path)
    X_test, y_test = load_tsfile_to_dataframe(test_path)
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    classes = np.unique(y)
    plt.figure(figsize=(8, 4))
    barycenters = []

    for c in classes:
        idx = np.where(y == c)[0]
        selected_idx = np.random.choice(idx, size=min(20, len(idx)), replace=False)
        samples = X[selected_idx]

        # Plot overlay
        for s in samples:
            plt.plot(s.ravel(), alpha=0.25)

        # Compute simple average barycenter
        barycenter = np.mean(np.vstack(samples), axis=0)
        barycenters.append(barycenter)
        plt.plot(barycenter, linewidth=3, label=f"Class {c} mean")

    # Compute DTW barycenter distance
    if len(barycenters) == 2:
        dtw_dist = dtw_distance(barycenters[0], barycenters[1])
        print(f"DTW barycenter distance: {dtw_dist:.4f}")
        results.append({"dataset": dataset, "DTW_distance": dtw_dist})
    else:
        results.append({"dataset": dataset, "DTW_distance": np.nan})

    plt.title(f"{dataset} – Raw Time-Series Overlay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset}_overlay.png"), dpi=300)
    plt.close()

# === SAVE SUMMARY TABLE ===
df = pd.DataFrame(results)
df.to_csv("EXPERIMENT_2B/DTW_summary_no_tslearn.csv", index=False)
print("\nSaved summary table to EXPERIMENT_2B/DTW_summary_no_tslearn.csv")
print(df)