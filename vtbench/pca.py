import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


# Configuration
DATASET_NAME = ["PhalangesOutlinesCorrect","ChlorineConcentration","SonyAIBORobotSurface1","Adiac","FaceAll","FacesUCR",
                "ArrowHead","CricketX","CricketY","CricketZ","InsectWingBeat","ToeSegmentation1","ToeSegmentation2",
                "Wine","WordSynonyms","Beef","BeetleFly","Computers","Earthquakes","Ham","Herring","RefrigerationDevices",
                "SharePriceIncrease","Crop","ECG5000","GunPoint","Lightning2","Strawberry","Yoga","FordB","Wafer"]
for d in DATASET_NAME:
    FEATURE_ROOT = f"{d}_results/single_chart"
    CHART_TYPES = ["line", "area", "bar", "scatter"]
    OUTPUT_DIR = f"{d}_results/intrinsic_dimensionality/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# PCA Analysis
def compute_intrinsic_dimensionality(X, var_threshold=0.9):
    """Return number of principal components explaining 90% (default) variance."""
    pca = PCA()
    pca.fit(X)
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.searchsorted(cumulative_var, var_threshold) + 1
    return n_components, cumulative_var


results = []

for chart_type in CHART_TYPES:
    feat_path = os.path.join(FEATURE_ROOT, chart_type, "features.npy")

    if not os.path.exists(feat_path):
        print(f"Missing features for {chart_type}")
        continue

    X = np.load(feat_path)
    X = X.reshape(X.shape[0], -1)
    print(f"Running PCA on {chart_type} features: {X.shape}")

    n_comp, cumulative_var = compute_intrinsic_dimensionality(X, var_threshold=0.9)
    results.append({"chart_type": chart_type, "n_components_90pct": n_comp})
    print(f"{chart_type}: {n_comp} components explain 90% of variance")

#resulting chart
if results:
    df = pd.DataFrame(results).set_index("chart_type")
    print("\n Intrinsic Dimensionality Summary:")
    print(df)

    # Heatmap visualization
    plt.figure(figsize=(5, 4))
    sns.heatmap(df.T, annot=True, cmap="coolwarm", cbar=False)
    plt.title(f"Intrinsic Dimensionality (90% Variance) – {DATASET_NAME}")
    plt.xlabel("Chart Type")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_pca_heatmap.png"), dpi=300)
    plt.show()

    print(f"\nSaved PCA plots to {OUTPUT_DIR}")
else:
    print("\nNo results — missing features for all chart types")
