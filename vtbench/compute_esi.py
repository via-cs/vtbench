import os
import subprocess
import pandas as pd
import numpy as np
import re

# config
datasets = ["PhalangesOutlinesCorrect","ChlorineConcentration","SonyAIBORobotSurface1","Adiac","FaceAll","FacesUCR",
                "ArrowHead","CricketX","CricketY","CricketZ","InsectWingBeat","ToeSegmentation1","ToeSegmentation2",
                "Wine","WordSynonyms","Beef","BeetleFly","Computers","Earthquakes","Ham","Herring","RefrigerationDevices",
                "SharePriceIncrease","Crop","ECG5000","GunPoint","Lightning2","Strawberry","Yoga","FordB","Wafer"]

# output directories
os.makedirs("results/cka", exist_ok=True)
os.makedirs("results/probe", exist_ok=True)
os.makedirs("results", exist_ok=True)

records = []


# Main pipeline
for dataset in datasets:
    print(f"\nProcessing dataset: {dataset}")

    cka_out = f"results/cka/{dataset}_cka.csv"

    # Run CKA extraction and capture ESI_CKA
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


df = pd.DataFrame(records)
df.to_csv("results/ESI_summary.csv", index=False)
print("\nSaved aggregated metrics to results/ESI_summary.csv")