import matplotlib.pyplot as plt
import numpy as np
import json

with open('best_settings_results_dict.json', 'r') as f:
    best_models_results_dict = json.load(f)

classes_to_include = [2, 3, 4]

fig, ax = plt.subplots(figsize=(12, 9))

class_colors = ['#d62728', '#9467bd', '#ff7f0e']  

n_samples = len(next(iter(best_models_results_dict.values()))['true_labels'])  # Total number of samples
bar_width = 6  
bar_padding = 2  
x_offsets = np.arange(len(best_models_results_dict))  

for i, (model_name, result) in enumerate(best_models_results_dict.items()):
    true_labels = np.array(result['true_labels'])
    predicted_labels = np.array(result['predicted_labels'])

  
    mask = np.isin(true_labels, classes_to_include)  
    true_labels = true_labels[mask]
    predicted_labels = predicted_labels[mask]

    # Sort samples based on their true class
    sorted_indices = np.argsort(true_labels)  
    true_labels = true_labels[sorted_indices]
    predicted_labels = predicted_labels[sorted_indices]

    n_samples_filtered = len(true_labels)
    
    y_positions = np.arange(n_samples_filtered)  

    for j in range(n_samples_filtered):
        color_index = classes_to_include.index(true_labels[j])  
        color = class_colors[color_index]  
        ax.hlines(y=y_positions[j], xmin=i * (bar_width + bar_padding), xmax=i * (bar_width + bar_padding) + bar_width,
                  colors=color, lw=1)  


    for j in range(n_samples_filtered):
        if true_labels[j] != predicted_labels[j]:
            ax.hlines(y=y_positions[j], xmin=i * (bar_width + bar_padding), 
                      xmax=i * (bar_width + bar_padding) + bar_width, colors='black', lw=1)
    

    class_boundaries = np.cumsum(np.bincount(true_labels - 2))  
    for boundary in class_boundaries[:-1]:  
        ax.axhline(y=boundary, color='black', linestyle='--', lw=0.8, zorder=10)


ax.set_title('Charts Types with Best Settings (Classes 3, 4, and 5)')
ax.set_xlabel('Model Type')
ax.set_ylabel('Instance')
ax.set_xticks(x_offsets * (bar_width + bar_padding) + bar_width / 2)
ax.set_xticklabels(best_models_results_dict.keys(), rotation=45, ha="right")
ax.set_yticks([])  


legend_labels = ['Class 3', 'Class 4', 'Class 5', 'Misclassified']
legend_handles = [plt.Line2D([0], [0], color=class_colors[i], lw=3) for i in range(len(class_colors))] + \
                 [plt.Line2D([0], [0], color='black', lw=3)]
ax.legend(legend_handles, legend_labels, loc='lower right', title="Legend")

plt.tight_layout()
plt.savefig('best_settings_viz_classes_grouped_3_4_5.png')
print("Plot saved as 'best_settings_viz_classes_grouped_3_4_5.png'")
