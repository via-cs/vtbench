import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tslearn.datasets import load_from_tsfile_to_dataframe
from tslearn.metrics import dtw
from tslearn.barycenters import dtw_barycenter_averaging


# === Configuration ===
data_root = "/Users/akkumy/Downloads/test_vtbench/vtbench/data" 
datasets = ["Beef", "BeetleFly"]    # datasets you want to test
output_dir = "results/raw_overlay"
os.makedirs(output_dir, exist_ok=True)

results = []

for dataset in datasets:
    print(f"\n=== Processing {dataset} ===")

    dataset_dir = os.path.join(data_root, dataset)
    train_path = os.path.join(dataset_dir, f"{dataset}_TRAIN.ts")
    test_path = os.path.join(dataset_dir, f"{dataset}_TEST.ts")

    # Load data (from .ts files in UCR format)
    X_train_df, y_train = load_from_tsfile_to_dataframe(train_path)
    X_test_df, y_test = load_from_tsfile_to_dataframe(test_path)

    # Convert DataFrame of series objects to numpy arrays
    X_train = np.stack(X_train_df.iloc[:, 0].apply(lambda x: np.array(x)).values)
    X_test = np.stack(X_test_df.iloc[:, 0].apply(lambda x: np.array(x)).values)

    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])


    classes = np.unique(y)
    print(f"Found classes: {classes}")

    plt.figure(figsize=(8, 4))
    barycenters = []

    for c in classes:
        idx = np.where(y == c)[0]
        selected_idx = np.random.choice(idx, size=min(20, len(idx)), replace=False)
        samples = X[selected_idx]

        # Plot overlay of samples
        for s in samples:
            plt.plot(s.ravel(), alpha=0.25)

        # Compute DTW barycenter
        barycenter = dtw_barycenter_averaging(samples)
        barycenters.append(barycenter)
        plt.plot(barycenter, linewidth=3, label=f"Class {c} mean")

    # Compute DTW barycenter distance (only for 2-class datasets)
    if len(barycenters) == 2:
        dtw_dist = dtw(barycenters[0].ravel(), barycenters[1].ravel())
        print(f"DTW barycenter distance: {dtw_dist:.4f}")
        results.append({"dataset": dataset, "DTW_distance": dtw_dist})
    else:
        print("Skipping DTW (dataset has more than 2 classes)")
        results.append({"dataset": dataset, "DTW_distance": np.nan})

    plt.title(f"{dataset} – Raw Time-Series Overlay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset}_overlay.png"), dpi=300)
    plt.close()

# === Save summary table ===
df = pd.DataFrame(results)
df.to_csv("EXPERIMENT_2B/DTW_summary.csv", index=False)
print("\nSaved summary table to EXPERIMENT_2B/DTW_summary.csv")
print(df)
