import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# =============================
# Configuration
# =============================
DATASET_NAME = "GunPoint"
FEATURE_ROOT = "GunPoint_results/single_chart"
CHART_TYPES = ["line", "area", "bar", "scatter"]
OUTPUT_DIR = "GunPoint_results/knn_probe_results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================
# Helper Functions
# =============================
def load_features(chart_type):
    feat_path = os.path.join(FEATURE_ROOT, chart_type, "features.npy")
    label_path = os.path.join(FEATURE_ROOT, chart_type, "labels.npy")
    if not (os.path.exists(feat_path) and os.path.exists(label_path)):
        raise FileNotFoundError(f"Missing features or labels for {chart_type}")
    X = np.load(feat_path)
    y = np.load(label_path)
    return X, y

def train_knn(X_train, y_train, k=5):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    return clf, scaler

def evaluate_knn(clf, scaler, X_test, y_test):
    X_test = scaler.transform(X_test)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc

# =============================
# Main experiment
# =============================
results = np.zeros((len(CHART_TYPES), len(CHART_TYPES)))

for i, train_chart in enumerate(CHART_TYPES):
    X_train, y_train = load_features(train_chart)
    clf, scaler = train_knn(X_train, y_train, k=5)

    for j, test_chart in enumerate(CHART_TYPES):
        X_test, y_test = load_features(test_chart)
        acc = evaluate_knn(clf, scaler, X_test, y_test)
        results[i, j] = acc
        print(f"KNN Train on {train_chart}; Test on {test_chart}: {acc:.3f}")

# =============================
# Visualization
# =============================
plt.figure(figsize=(7,6))
sns.heatmap(results, annot=True, fmt=".2f", cmap="viridis",
            xticklabels=CHART_TYPES, yticklabels=CHART_TYPES,
            cbar_kws={'label': 'KNN Accuracy'})
plt.title(f"KNN Cross-Chart Transfer – {DATASET_NAME}", fontsize=14)
plt.xlabel("Test Chart Type")
plt.ylabel("Train Chart Type")

out_path = os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_knn_transfer_heatmap.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.show()

print(f"\n Saved heatmap to {out_path}")
