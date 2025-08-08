# 📊 VTBench  
*A Framework for Chart-Based & Multimodal Time-Series Classification*

- VTBench is a **Python package** for time-series classification using **visual chart representations** (line, bar, area, scatter) — individually or fused with raw numerical features.  
- It provides a **modular, extensible framework** for building, training, and evaluating both unimodal and multimodal architectures.

<img width="2318" height="952" alt="image" src="https://github.com/user-attachments/assets/d4bf0cc9-a5fa-40d2-b273-f78c64c79862" />



## Highlights
- Auto-generate **4 chart types** from time series  
- Compare **single-modal, multi-chart, and multimodal** setups  
- Supports **CNN, Transformer, OS-CNN** for different modalities  
- Flexible **fusion strategies** (concat, weighted)  
- Built-in **UCR Archive dataset support**  
- Config-driven & reproducible experiments  
- Extensible to new architectures, fusion strategies, and modalities 


## Installation
Install in editable mode with modern Python packaging:
```bash
git clone https://github.com/<your-username>/vtbench.git
cd vtbench
pip install -e .
```
This command will:

- Parse the YAML config
- Generate required chart images from the time-series data
- Load data and initialize the appropriate model
- Train and evaluate the model
- Save results to the results/ folder

## Configuration Files

VTBench is driven by YAML configuration files that define model type, data paths, training parameters, fusion strategies and more. You can run VTBench with any of the following configs using:

```bash
vtbench --config vtbench/config/<config_file>.yaml
```

## Available Configs

####  `single_modal_chart.yaml`
Runs a single chart-type image model (e.g., line, bar, etc.) using a CNN. Useful for testing how well visual representations alone perform on classification tasks.

#### `two_branch.yaml`
Combines chart image input with raw numerical data in a two-branch architecture. Helps compare unimodal vs. multimodal performance with a simple fusion strategy.

####  `multi_modal_chart.yaml`
Uses multiple chart types (e.g., line, bar, scatter, area) in parallel branches. Demonstrates how different visual representations can complement each other in a multimodal structure.


## Use
VTBench is designed for:
- Benchmarking chart-based vs. raw numerical TSC
- Ablation studies on chart types, fusion strategies, and architectures
- Reproducible research via config-driven design
- Extending to new encoders (ResNet, ViT, LSTM, TCN, etc.) and modalities (spectrograms, text metadata, etc.)
