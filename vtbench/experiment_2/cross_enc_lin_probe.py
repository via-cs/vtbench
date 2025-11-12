import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
from extract_features_and_cka import mean_offdiag


def load_features(chart_type, feature_root):
    """Load pre-extracted features and labels for a chart type."""
    feat_path = os.path.join(feature_root, chart_type, "features.npy")
    label_path = os.path.join(feature_root, chart_type, "labels.npy")

    print(f"feature path : {feat_path}, label path: {label_path}")

    if not (os.path.exists(feat_path) and os.path.exists(label_path)):
        raise FileNotFoundError(f"Missing features or labels for {chart_type} in {feature_root}")

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
    """Evaluate trained probe on another feature set."""
    X_test = scaler.transform(X_test)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc


def main(dataset_name, output_path=None):
    # === Configuration ===
    dname = f"{dataset_name}_results"
    feature_root = os.path.join(dname, "single_chart")
    chart_types = ["line", "area", "bar", "scatter"]
    output_dir = os.path.join(dataset_name, "linear_probe_results")
    os.makedirs(output_dir, exist_ok=True)

    # === Cross-encoding probe evaluation ===
    results = np.zeros((len(chart_types), len(chart_types)))

    for i, train_chart in enumerate(chart_types):
        X_train, y_train = load_features(train_chart, feature_root)
        clf, scaler = train_linear_probe(X_train, y_train)

        for j, test_chart in enumerate(chart_types):
            X_test, y_test = load_features(test_chart, feature_root)
            acc = evaluate_probe(clf, scaler, X_test, y_test)
            results[i, j] = acc
            print(f"Train on {train_chart} → Test on {test_chart}: {acc:.3f}")

    # === Save numeric results ===
    #results_df = pd.DataFrame(results, index=chart_types, columns=chart_types)
    #out_csv = output_path or os.path.join(output_dir, f"{dataset_name}_probe.csv")
    #results_df.to_csv(out_csv, index=True)
    #print(f"\nSaved probe matrix to {out_csv}")

    # === Compute Encoding Sensitivity Index (ESI_PROBE) ===
    try:
        esi_probe = 1 - mean_offdiag(results)
        print(f"ESI_PROBE={esi_probe:.6f}")
    except Exception as e:
        print(f"Could not compute ESI_PROBE: {e}")
    # === Visualization ===
    plt.figure(figsize=(7,6))
    sns.heatmap(
        results,
        annot=True,
        fmt=".2f",
        cmap="magma",
        xticklabels=chart_types,
        yticklabels=chart_types,
        cbar_kws={'label': 'Accuracy'}
    )
    plt.title(f"Cross-Chart Linear Probe – {dataset_name}", fontsize=14)
    plt.xlabel("Test Chart Type")
    plt.ylabel("Train Chart Type")

    heatmap_out = os.path.join(output_dir, f"{dataset_name}_linear_probe_heatmap.png")
    plt.tight_layout()
    plt.savefig(heatmap_out, dpi=300)
    plt.close()

    print(f"Saved heatmap to {heatmap_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cross-encoding linear probe across chart types")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., CricketY_results)")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save probe CSV")
    args = parser.parse_args()

    main(args.dataset, args.output)
