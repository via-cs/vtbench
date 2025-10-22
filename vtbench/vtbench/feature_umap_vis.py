import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import pandas as pd

# =========================
# Configuration
# =========================
DATASET_NAME = "GunPoint"  # change per dataset
FEATURE_ROOT = f"{DATASET_NAME}_results/single_chart"
CHART_TYPES = ["line", "area", "bar", "scatter"]
OUTPUT_DIR = f"{DATASET_NAME}_results/umap_visualization/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Load and combine features
# =========================
all_features = []
all_labels = []
all_chart_labels = []

for chart_type in CHART_TYPES:
    feat_path = os.path.join(FEATURE_ROOT, chart_type, "features.npy")
    label_path = os.path.join(FEATURE_ROOT, chart_type, "labels.npy")

    if not (os.path.exists(feat_path) and os.path.exists(label_path)):
        print(f"Missing features for {chart_type}, skipping.")
        continue

    X = np.load(feat_path)
    y = np.load(label_path)

    all_features.append(X)
    all_labels.append(y)
    all_chart_labels.append(np.array([chart_type] * len(y)))

if not all_features:
    print("No features found for any chart type.")
    exit()

# Combine all into single arrays
X_all = np.concatenate(all_features, axis=0)
y_all = np.concatenate(all_labels, axis=0)
chart_all = np.concatenate(all_chart_labels, axis=0)

print(f"Loaded combined feature set: {X_all.shape}")

# =========================
# Run UMAP
# =========================
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
X_2d = reducer.fit_transform(X_all)

# =========================
# Create a DataFrame for plotting
# =========================
df = pd.DataFrame({
    "x": X_2d[:, 0],
    "y": X_2d[:, 1],
    "label": y_all,
    "chart_type": chart_all
})

# =========================
# Plot 1 – Color by true class
# =========================
plt.figure(figsize=(7, 6))
sns.scatterplot(
    data=df, x="x", y="y", hue="label", style="chart_type",
    palette="Spectral", s=25, alpha=0.8
)
plt.title(f"UMAP Projection by Class – {DATASET_NAME}", fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Class")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_umap_by_class.png"), dpi=300)
plt.show()

# =========================
# Plot 2 – Color by chart type
# =========================
plt.figure(figsize=(7, 6))
sns.scatterplot(
    data=df, x="x", y="y", hue="chart_type", palette="tab10", s=25, alpha=0.8
)
plt.title(f"UMAP Projection by Chart Type – {DATASET_NAME}", fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Chart Type")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_umap_by_chart.png"), dpi=300)
plt.show()

print(f"\n Saved UMAP visualizations to {OUTPUT_DIR}")
