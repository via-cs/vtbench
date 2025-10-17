import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# Conf
DATASET_NAME = "GunPoint"   # change per dataset
FEATURE_ROOT = "GunPoint_results/single_chart"  # base folder
CHART_TYPES = ["line", "area", "bar", "scatter"]
OUTPUT_DIR = "GunPoint_results/linear_probe_results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Helper funcs
def load_features(chart_type):
    """Load pre-extracted features and labels."""
    feat_path = os.path.join(FEATURE_ROOT, chart_type, "features.npy")
    label_path = os.path.join(FEATURE_ROOT, chart_type, "labels.npy")
    if not (os.path.exists(feat_path) and os.path.exists(label_path)):
        raise FileNotFoundError(f"Missing features or labels for {chart_type}")
    X = np.load(feat_path)
    y = np.load(label_path)
    return X, y

def train_linear_probe(X_train, y_train):
    """Train a simple logistic regression classifier."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    return clf, scaler

def evaluate_probe(clf, scaler, X_test, y_test):
    """Evaluate the trained probe on another feature set."""
    X_test = scaler.transform(X_test)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc


# Main implementation
results = np.zeros((len(CHART_TYPES), len(CHART_TYPES)))

for i, train_chart in enumerate(CHART_TYPES):
    X_train, y_train = load_features(train_chart)
    clf, scaler = train_linear_probe(X_train, y_train)

    for j, test_chart in enumerate(CHART_TYPES):
        X_test, y_test = load_features(test_chart)
        acc = evaluate_probe(clf, scaler, X_test, y_test)
        results[i, j] = acc
        print(f"Train on {train_chart} → Test on {test_chart}: {acc:.3f}")


# Visualization
plt.figure(figsize=(7,6))
sns.heatmap(results, annot=True, fmt=".2f", cmap="magma", 
            xticklabels=CHART_TYPES, yticklabels=CHART_TYPES, cbar_kws={'label': 'Accuracy'})
plt.title(f"Cross-Chart Linear Probe – {DATASET_NAME}", fontsize=14)
plt.xlabel("Test Chart Type")
plt.ylabel("Train Chart Type")

out_path = os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_linear_probe_heatmap.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.show()

print(f"\n Saved heatmap to {out_path}")
