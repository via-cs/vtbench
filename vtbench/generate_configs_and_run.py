import os
import subprocess

# classes for each dataset
num_classes_map = {
    "PhalangesOutlinesCorrect": 2,
    "ChlorineConcentration":    3,
    "SonyAIBORobotSurface1":    2,
    "Adiac":                    37,
    "FaceAll":                  14,
    "FacesUCR":                 14,
    "ArrowHead":                3,
    "CricketX":                 12,
    "CricketY":                 12,
    "CricketZ":                 12,
    "InsectWingBeat":           11,
    "ToeSegmentation1":         2,
    "ToeSegmentation2":         2,
    "Wine":                     2,
    "WordSynonyms":             25,
    "Beef":                     5,
    "BeetleFly":                2,
    "Computers":                2,
    "Earthquakes":              2,
    "Ham":                      2,
    "Herring":                  2,
    "RefrigerationDevices":     3,
    "SharePriceIncrease":       2,
    "Crop":                     24,
    "ECG5000":                  5,
    "GunPoint":                 2,
    "Lightning2":               2,
    "Strawberry":               2,
    "Yoga":                     2,
    "FordB":                    2,
    "Wafer":                    2,
    "InsectWingBeat":           10
}

# Base YAML template
template = """dataset:
  name: {dataset}
  train_path: /vtbench/data/{dataset}/{dataset}_TRAIN.ts
  test_path: /vtbench/data/{dataset}/{dataset}_TEST.ts
image_generation:
  generate_images: true
  overwrite_existing: true
model:
  type: single_modal_chart
  chart_model: deepcnn
  input_channels: 3
  num_classes: {num_classes}
chart_branches:
  branch_1:
    chart_type: {chart}
    color_mode: color
    label_mode: with_label
training:
  batch_size: 32
  epochs: 10
  learning_rate: 0.0005
output:
  dir: {dataset}_results/single_chart/{chart}/
  save_model: true
"""

# Datasets to generate configs and train for
datasets = ["PhalangesOutlinesCorrect","ChlorineConcentration","SonyAIBORobotSurface1","Adiac","FaceAll","FacesUCR",
                "ArrowHead","CricketX","CricketY","CricketZ","InsectWingBeat","ToeSegmentation1","ToeSegmentation2",
                "Wine","WordSynonyms","Beef","BeetleFly","Computers","Earthquakes","Ham","Herring","RefrigerationDevices",
                "SharePriceIncrease","Crop","ECG5000","GunPoint","Lightning2","Strawberry","Yoga","FordB","Wafer"]
chart_types = ["line", "bar", "area", "scatter"]

# Output directory
output_dir = "config/generated_config"
os.makedirs(output_dir, exist_ok=True)

for dataset in datasets:
    if dataset not in num_classes_map:
        print(f"[SKIP] {dataset} — not in num_classes_map, add it first.")
        continue

    num_classes = num_classes_map[dataset]

    for chart in chart_types:
        config_filename = f"{dataset}_{chart}.yaml"
        config_path = os.path.join(output_dir, config_filename)

        with open(config_path, "w") as f:
            f.write(template.format(
                dataset=dataset,
                chart=chart,
                num_classes=num_classes,
            ))
        print(f"Created config: {config_path}  (num_classes={num_classes})")

        # Run training
        subprocess.run(["python", "main.py", "--config", config_path])