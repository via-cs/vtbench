import os
import subprocess
import pandas as pd
import numpy as np
import re

# === Configuration ===
datasets = [
   "Adiac",
   "ArrowHead",
   "Beef",
   "BeetleFly",
   "ChlorineConcentration",
   "Computers",
   "CricketX",
   "CricketY",
   "CricketZ",
   "ECG5000",
   "GunPoint",
   "Yoga",
   "Strawberry",
   "FordB",
   "Earthquakes",
    "SonyAIBORobotSurface1",
    "ToeSegmentation1",
    "ToeSegmentation2",
    "Wine",
    "WordSynonyms",
    "Ham",
    "Herring",
    "Lightning2",
    "PhalangesOutlinesCorrect",
    "RefrigerationDevices",
    "SharePriceIncrease",
    "FaceAll",
    "Crop",
    "Wafer"

]

# Define output directories
os.makedirs("results/cka", exist_ok=True)
os.makedirs("results/probe", exist_ok=True)
os.makedirs("results", exist_ok=True)

records = []


# === Main pipeline ===
for dataset in datasets:
    print(f"\n=== Processing dataset: {dataset} ===")

    cka_out = f"results/cka/{dataset}_cka.csv"

    # --- Step 1 + 2: Run CKA extraction and capture ESI_CKA ---
    print("-> Running CKA feature extraction...")
    result_cka = subprocess.run(
        ["python", "extract_features_and_cka.py", "--dataset", dataset, "--output", cka_out],
        capture_output=True, text=True
    )

    esi_cka = None
    match_cka = re.search(r"ESI_CKA=([0-9.]+)", result_cka.stdout)
    if match_cka:
        esi_cka = float(match_cka.group(1))
        print(f"ESI_CKA for {dataset}: {esi_cka:.4f}")
    else:
        print(f"Could not find ESI_CKA output for {dataset}")
        print(result_cka.stdout)

    # --- Step 3: Run Probe evaluation and capture ESI_PROBE ---
    print("-> Running linear probe evaluation...")
    result_probe = subprocess.run(
        ["python", "cross_enc_lin_probe.py", "--dataset", dataset],
        capture_output=True, text=True
    )

    esi_probe = None
    match_probe = re.search(r"ESI_PROBE=([0-9.]+)", result_probe.stdout)
    if match_probe:
        esi_probe = float(match_probe.group(1))
        print(f"ESI_PROBE for {dataset}: {esi_probe:.4f}")
    else:
        print(f"Could not find ESI_PROBE output for {dataset}")
        print(result_probe.stdout)

    records.append({
        "dataset": dataset,
        "ESI_CKA": esi_cka,
        "ESI_PROBE": esi_probe
    })

    print(f"Completed {dataset}")


# === Step 4: Save aggregated metrics ===
df = pd.DataFrame(records)
df.to_csv("results/ESI_summary.csv", index=False)
print("\nSaved aggregated metrics to results/ESI_summary.csv")

# Optional: visualize
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    df_sorted_cka = df.sort_values("ESI_CKA")
    axes[0].bar(df_sorted_cka["dataset"], df_sorted_cka["ESI_CKA"])
    axes[0].set_title("Encoding Sensitivity Index (ESI_CKA)")
    axes[0].set_ylabel("ESI_CKA")
    axes[0].tick_params(axis="x", rotation=45)

    df_sorted_probe = df.sort_values("ESI_PROBE")
    axes[1].bar(df_sorted_probe["dataset"], df_sorted_probe["ESI_PROBE"])
    axes[1].set_title("Encoding Sensitivity Index (ESI_PROBE)")
    axes[1].set_ylabel("ESI_PROBE")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("results/ESI_summary_bar.png")
    plt.close()
    print("Saved bar charts to results/ESI_summary_bar.png")
except ImportError:
    print("matplotlib not installed — skipping plot generation.")


import matplotlib.pyplot as plt

# ========= Dataset Class Splits (from your screenshot) =========
binary_datasets = [
    "FordB", "Wine", "Herring", "SharePriceIncrease"
    "StrawBerry", "PhalangesOutlinesCorrect",
    "Wafer", "Ham", "Beetlefly", "Earthquakes",
    "Yoga", "GunPoint", "Lightning2",
    "ToeSegmentation1", "ToeSegmentation2",
    "SonyAIBORobotSurface1", "Computers"
]

multiclass_datasets = [
    "Adiac", "Beef", "ChlorineConcentration",
    "Crop", "FaceAll",
    "ECG5000", "CricketY", "InsectWingbeatSound",
    "Arrowhead", "CricketX", "CricketZ",
    "FacesUCR", "RefrigerationDevices", "WordSynonyms"
]

# ========= FULL DATASET PLOTS (CKA + PROBE) =========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df_sorted_cka = df.sort_values("ESI_CKA")
axes[0].bar(df_sorted_cka["dataset"], df_sorted_cka["ESI_CKA"])
axes[0].set_title("ESI_CKA for All Datasets")
axes[0].set_ylabel("ESI_CKA")
axes[0].tick_params(axis="x", rotation=60)

df_sorted_probe = df.sort_values("ESI_PROBE")
axes[1].bar(df_sorted_probe["dataset"], df_sorted_probe["ESI_PROBE"])
axes[1].set_title("ESI_PROBE for All Datasets")
axes[1].set_ylabel("ESI_PROBE")
axes[1].tick_params(axis="x", rotation=60)

plt.tight_layout()
plt.savefig("results/ESI_all_datasets.png")
plt.close()


# ========= BINARY & MULTICLASS SUBSETS =========

df_binary = df[df["dataset"].isin(binary_datasets)]
df_multiclass = df[df["dataset"].isin(multiclass_datasets)]

# --- Sort for cleaner plotting ---
df_binary = df_binary.sort_values("ESI_CKA")
df_multiclass = df_multiclass.sort_values("ESI_CKA")

# ========= CKA: Binary vs Multiclass =========
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

axes2[0].bar(df_binary["dataset"], df_binary["ESI_CKA"])
axes2[0].set_title("Binary Datasets – ESI_CKA")
axes2[0].set_ylabel("ESI_CKA")
axes2[0].tick_params(axis="x", rotation=60)

axes2[1].bar(df_multiclass["dataset"], df_multiclass["ESI_CKA"])
axes2[1].set_title("Multiclass Datasets – ESI_CKA")
axes2[1].set_ylabel("ESI_CKA")
axes2[1].tick_params(axis="x", rotation=60)

plt.tight_layout()
plt.savefig("results/ESI_CKA_binary_vs_multiclass.png")
plt.close()


# ========= PROBE: Binary vs Multiclass =========
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

df_binary_probe = df_binary.sort_values("ESI_PROBE")
df_multiclass_probe = df_multiclass.sort_values("ESI_PROBE")

axes3[0].bar(df_binary_probe["dataset"], df_binary_probe["ESI_PROBE"])
axes3[0].set_title("Binary Datasets – ESI_PROBE")
axes3[0].set_ylabel("ESI_PROBE")
axes3[0].tick_params(axis="x", rotation=60)

axes3[1].bar(df_multiclass_probe["dataset"], df_multiclass_probe["ESI_PROBE"])
axes3[1].set_title("Multiclass Datasets – ESI_PROBE")
axes3[1].set_ylabel("ESI_PROBE")
axes3[1].tick_params(axis="x", rotation=60)

plt.tight_layout()
plt.savefig("results/ESI_PROBE_binary_vs_multiclass.png")
plt.close()

print("Saved:\n"
      "- ESI_all_datasets.png\n"
      "- ESI_CKA_binary_vs_multiclass.png\n"
      "- ESI_PROBE_binary_vs_multiclass.png")

