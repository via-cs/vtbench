import os
import subprocess
import pandas as pd
import numpy as np
import re

# === Configuration ===
datasets = [
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
    "Strawberry",
    "FordB",
    "Yoga"
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
