import matplotlib.pyplot as plt
import numpy as np
import json

with open('best_settings_results_dict.json', 'r') as f:
    best_models_results_dict = json.load(f)

fig, ax = plt.subplots(figsize=(12, 20)) 

n_samples = len(next(iter(best_models_results_dict.values()))['true_labels'])  

bar_width = 6  
bar_padding = 2  
x_offsets = np.arange(len(best_models_results_dict)) 

class_colors = ['#1f77b4', '#2ca02c', '#d674cc', '#9467bd', '#ff7f0e']


true_labels = np.array(next(iter(best_models_results_dict.values()))['true_labels'])
unique_labels, class_counts = np.unique(true_labels, return_counts=True)


class_boundaries = np.cumsum(class_counts)

line_gap = 0.8
vertical_offset = 5

for i, (model_name, result) in enumerate(best_models_results_dict.items()):
    true_labels = np.array(result['true_labels'])
    predicted_labels = np.array(result['predicted_labels'])

    class_indices = np.argsort(true_labels)
    true_labels = true_labels[class_indices]
    predicted_labels = predicted_labels[class_indices]
   
    
    y_positions = np.arange(n_samples) * (1 + line_gap)  

    for j in range(n_samples):
        color = class_colors[true_labels[j]]  
        ax.hlines(y=y_positions[j], xmin=i * (bar_width + bar_padding), xmax=i * (bar_width + bar_padding) + bar_width,
                  colors=color, lw=1.0) 

    for j in range(n_samples):
        if true_labels[j] != predicted_labels[j]:
            ax.hlines(y=y_positions[j] - vertical_offset, xmin=i * (bar_width + bar_padding), 
                      xmax=i * (bar_width + bar_padding) + bar_width, colors='black', lw=1.0)

    for boundary in class_boundaries:
        boundary_y_position = boundary * (1 + line_gap)
        ax.axhline(y=boundary_y_position, color='black', linestyle='--', lw=0.8, zorder=10)


ax.set_title('Charts Types with Best Settings(Grouped by Classes)')
ax.set_xlabel('Model Type')
ax.set_ylabel('Instance')
ax.set_xticks(x_offsets * (bar_width + bar_padding) + bar_width / 2)
ax.set_xticklabels(best_models_results_dict.keys(), rotation=45, ha="right")
ax.set_yticks([]) 


legend_labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Misclassified']
legend_handles = [plt.Line2D([0], [0], color=class_colors[i], lw=2) for i in range(len(class_colors))] + \
                 [plt.Line2D([0], [0], color='black', lw=2)]
ax.legend(legend_handles, legend_labels, loc='lower right', title="Legend")
plt.tight_layout()

plt.savefig('best_settings_viz.png')
print("Plot saved as 'best_settings_viz.png'")
