import matplotlib.pyplot as plt
import numpy as np
import json
import yaml
import os

def load_config(config_path):
    """Load the YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def generate_visualization(dataset_name, json_file):
    """Generate a visualization for a given dataset and JSON file."""

    with open(json_file, 'r') as f:
        best_models_results_dict = json.load(f)

    fig, ax = plt.subplots(figsize=(12, 12))

    n_samples = len(next(iter(best_models_results_dict.values()))['true_labels'])

    bar_width = 6
    bar_padding = 2
    x_offsets = np.arange(len(best_models_results_dict))  
    line_gap = 0.8
    vertical_offset = 5

    class_colors = ['#1f77b4', '#ff7f0e']  # Blue for Class 0, Orange for Class 1

    for i, (model_name, result) in enumerate(best_models_results_dict.items()):
        true_labels = np.array(result['true_labels'])
        predicted_labels = np.array(result['predicted_labels'])

        class_indices = np.argsort(true_labels)
        true_labels = true_labels[class_indices]
        predicted_labels = predicted_labels[class_indices]

        y_positions = np.arange(n_samples) * (1 + line_gap)  

        # Plot true labels
        for j in range(n_samples):
            color = class_colors[true_labels[j]]  
            ax.hlines(y=y_positions[j], xmin=i * (bar_width + bar_padding), 
                      xmax=i * (bar_width + bar_padding) + bar_width,
                      colors=color, lw=1.0)

        # Plot misclassified labels
        for j in range(n_samples):
            if true_labels[j] != predicted_labels[j]:
                ax.hlines(y=y_positions[j] - vertical_offset, xmin=i * (bar_width + bar_padding), 
                          xmax=i * (bar_width + bar_padding) + bar_width, colors='black', lw=1.0)

    ax.set_title(f'Visualization of Model Predictions for {dataset_name}')
    ax.set_xlabel('Model Type')
    ax.set_ylabel('Samples')
    ax.set_xticks(x_offsets * (bar_width + bar_padding) + bar_width / 2)
    ax.set_xticklabels(best_models_results_dict.keys(), rotation=45, ha="right")
    ax.set_yticks([])

    legend_labels = ['Class 0', 'Class 1', 'Misclassified']
    legend_handles = [
        plt.Line2D([0], [0], color=class_colors[0], lw=2),
        plt.Line2D([0], [0], color=class_colors[1], lw=2),
        plt.Line2D([0], [0], color='black', lw=2)
    ]
    ax.legend(legend_handles, legend_labels, loc='lower right', title="Legend")

    output_file = (f"{dataset_name}_viz.png")
    plt.tight_layout()
    plt.savefig(output_file)

    print(f"Visualization saved to '{output_file}'")
    

def main(config_path):
    """Main function to generate visualizations for datasets."""
    config = load_config(config_path)

    for dataset in config['datasets']:
        dataset_name = dataset['name']
        json_file = f"{dataset_name}_results.json"
        

        print(f"Generating visualization for dataset: {dataset_name}")
        generate_visualization(dataset_name, json_file)


if __name__ == "__main__":
    config_path = "config_viz.yaml"  
    main(config_path)
