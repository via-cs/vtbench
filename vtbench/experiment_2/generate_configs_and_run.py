import os
import subprocess

# base YAML template
template = """dataset:
  name: {dataset}
  train_path: /Users/akkumy/Downloads/test_vtbench/vtbench/data/{dataset}/{dataset}_TRAIN.ts
  test_path: /Users/akkumy/Downloads/test_vtbench/vtbench/data/{dataset}/{dataset}_TEST.ts

image_generation:
  generate_images: true
  overwrite_existing: true

model:
  type: single_modal_chart
  chart_model: deepcnn
  input_channels: 3  
  num_classes: 2

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

# dataset list (from your screenshot)
'''datasets = ["Crop", "Earthquakes", "FaceAll",
    "FacesUCR", "Ham", "Herring", "InsectWingbeat", "Lightning2",
    "PhalangesOutlinesCorrect", "RefrigerationDevices", "SharePriceIncrease", 
    "SonyAIBORobotSurface1", "ToeSegmentation1", "ToeSegmentation2", 
    "Wafer", "Wine", "WordSynonyms"]'''

datasets = ["Strawberry", "GunPoint", "Yoga"]
chart_types = ["line", "scatter", "area", "bar"]

# output directory for generated configs
output_dir = "config/generated_configs"
os.makedirs(output_dir, exist_ok=True)

# loop and generate
for dataset in datasets:
    for chart in chart_types:
        config_filename = f"{dataset}_{chart}.yaml"
        config_path = os.path.join(output_dir, config_filename)
        
        # write YAML file
        with open(config_path, "w") as f:
            f.write(template.format(dataset=dataset, chart=chart))
        
        print(f" Created config: {config_path}")

        # automatically execute training
        subprocess.run(["python", "main.py", "--config", config_path])
