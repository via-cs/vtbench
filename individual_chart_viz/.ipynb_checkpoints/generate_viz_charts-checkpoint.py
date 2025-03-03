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

    json_file_path = os.path.join("individual_chart_viz", "json_dicts", json_file)

    if not os.path.exists(json_file_path):
        print(f"Error: JSON file '{json_file_path}' not found.")
        return

    with open(json_file_path, 'r') as f:
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

        accuracy = np.mean(true_labels == predicted_labels) * 100

        class_indices = np.argsort(true_labels)
        true_labels = true_labels[class_indices]
        predicted_labels = predicted_labels[class_indices]

        y_positions = np.arange(n_samples) * (1 + line_gap)  

        for j in range(n_samples):
            color = class_colors[true_labels[j]]  
            ax.hlines(y=y_positions[j], xmin=i * (bar_width + bar_padding), 
                      xmax=i * (bar_width + bar_padding) + bar_width,
                      colors=color, lw=1.0)

        for j in range(n_samples):
            if true_labels[j] != predicted_labels[j]:
                ax.hlines(y=y_positions[j] - vertical_offset, xmin=i * (bar_width + bar_padding), 
                          xmax=i * (bar_width + bar_padding) + bar_width, colors='black', lw=1.0)

        ax.text(i * (bar_width + bar_padding) + bar_width / 2, max(y_positions) + 5,
                f'{accuracy:.2f}%', ha='center', fontsize=10, color='green')

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


def generate_visualization_with_merged_bar(dataset_name, json_file):
    """Generate a visualization with a merged performance bar for a given dataset and JSON file."""

    json_file_path = os.path.join("individual_chart_viz", "json_dicts", json_file)

    if not os.path.exists(json_file_path):
        print(f"Error: JSON file '{json_file_path}' not found.")
        return

    with open(json_file_path, 'r') as f:
        best_models_results_dict = json.load(f)

    fig, ax = plt.subplots(figsize=(12, 12))

    n_samples = len(next(iter(best_models_results_dict.values()))['true_labels'])

    bar_width = 6
    bar_padding = 2
    x_offsets = np.arange(len(best_models_results_dict) + 1)  
    line_gap = 0.8
    vertical_offset = 5

    class_colors = ['#1f77b4', '#ff7f0e']  

    all_true_labels = None
    merged_correct = np.zeros(n_samples, dtype=bool)

    for i, (model_name, result) in enumerate(best_models_results_dict.items()):
        true_labels = np.array(result['true_labels'])
        predicted_labels = np.array(result['predicted_labels'])

        if all_true_labels is None:
            all_true_labels = true_labels 
        else:
            if not np.array_equal(all_true_labels, true_labels):
                print(f"Error: True labels mismatch between charts. Ensure all charts use the same dataset.")
                return

       
        merged_correct |= (true_labels == predicted_labels)

        
        class_0_indices = np.where(true_labels == 0)[0]
        class_1_indices = np.where(true_labels == 1)[0]
        sorted_indices = np.concatenate([class_0_indices, class_1_indices])
        
        true_labels = true_labels[sorted_indices]
        predicted_labels = predicted_labels[sorted_indices]
        
        
        accuracy = np.mean(true_labels == predicted_labels) * 100
        y_positions = np.arange(len(true_labels)) * (1 + line_gap)
        ax.text(i * (bar_width + bar_padding) + bar_width / 2, max(y_positions) + 10, f'{accuracy:.2f}%', ha='center', fontsize=10, color='green')

        for j in range(len(true_labels)):
            color = class_colors[true_labels[j]]
            ax.hlines(y=y_positions[j], xmin=i * (bar_width + bar_padding),
                      xmax=i * (bar_width + bar_padding) + bar_width,
                      colors=color, lw=1.0)

        for j in range(len(true_labels)):
            if true_labels[j] != predicted_labels[j]:
                ax.hlines(y=y_positions[j] - vertical_offset, xmin=i * (bar_width + bar_padding),
                          xmax=i * (bar_width + bar_padding) + bar_width, colors='black', lw=1.0)

   
    all_true_labels = all_true_labels[np.concatenate([np.where(all_true_labels == 0)[0], np.where(all_true_labels == 1)[0]])]
    merged_correct = merged_correct[np.concatenate([np.where(all_true_labels == 0)[0], np.where(all_true_labels == 1)[0]])]

    y_positions = np.arange(len(all_true_labels)) * (1 + line_gap)
    for j in range(len(all_true_labels)):
        if merged_correct[j]:
            color = class_colors[all_true_labels[j]]
            ax.hlines(y=y_positions[j], xmin=len(best_models_results_dict) * (bar_width + bar_padding),
                      xmax=len(best_models_results_dict) * (bar_width + bar_padding) + bar_width,
                      colors=color, lw=1.0)
        else:
            ax.hlines(y=y_positions[j] - vertical_offset, xmin=len(best_models_results_dict) * (bar_width + bar_padding),
                      xmax=len(best_models_results_dict) * (bar_width + bar_padding) + bar_width, colors='black', lw=1.0)

    merged_accuracy = np.mean(merged_correct) * 100
    ax.text(len(best_models_results_dict) * (bar_width + bar_padding) + bar_width / 2,
            max(y_positions) + 10,
            f'{merged_accuracy:.2f}%', ha='center', fontsize=10, color='green')

    ax.set_title(f'Visualization of Model Predictions for {dataset_name} with Merged Bar')
    ax.set_xlabel('Model Type')
    ax.set_ylabel('Samples')
    ax.set_xticks(x_offsets * (bar_width + bar_padding) + bar_width / 2)
    ax.set_xticklabels(list(best_models_results_dict.keys()) + ['Merged'], rotation=45, ha="right")
    ax.set_yticks([])

    legend_labels = ['Class 0', 'Class 1', 'Misclassified']
    legend_handles = [
        plt.Line2D([0], [0], color=class_colors[0], lw=2),
        plt.Line2D([0], [0], color=class_colors[1], lw=2),
        plt.Line2D([0], [0], color='black', lw=2)
    ]
    ax.legend(legend_handles, legend_labels, loc='lower right', title="Legend")

    output_file = (f"{dataset_name}_viz_with_merged_bar.png")
    plt.tight_layout()
    plt.savefig(output_file)

    print(f"Visualization with merged bar saved to '{output_file}'")



def main(config_path):
    """Main function to generate visualizations for datasets."""
    config = load_config(config_path)

    for dataset in config['datasets']:
        dataset_name = dataset['name']
        json_file = f"{dataset_name}_results.json"

        print(f"Generating visualization for dataset: {dataset_name}")
        generate_visualization_with_merged_bar(dataset_name, json_file)


if __name__ == "__main__":
    config_path = "config_viz.yaml"  
    main(config_path)
