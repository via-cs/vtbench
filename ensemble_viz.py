import matplotlib.pyplot as plt
import numpy as np
import json

# Load the ensemble results
with open('ensemble_results_dict.json', 'r') as f:
    ensemble_results_dict = json.load(f)

# Access ensemble data
ensemble_data = ensemble_results_dict["Ensemble Majority Voting"]
true_labels = np.array(ensemble_data["true_labels"])
predicted_labels = np.array(ensemble_data["predicted_labels"])

# Sort data by true labels for grouping by class
sorted_indices = np.argsort(true_labels)
true_labels = true_labels[sorted_indices]
predicted_labels = predicted_labels[sorted_indices]

# Check unique values to confirm they are 0-4
unique_labels = np.unique(true_labels)
print("Unique labels in true_labels:", unique_labels)  # Should print [0 1 2 3 4]

# Set up plot parameters
fig, ax = plt.subplots(figsize=(6, 20))  # Increased height for better visibility
n_samples = len(true_labels)
bar_width = 2  # Adjust the bar width if needed
y_positions = np.arange(n_samples)  # Position for each line in the chart

# Define colors for classes 0 through 4
class_colors = ['#1f77b4', '#2ca02c', '#d674cc', '#9467bd', '#ff7f0e']  # Blue, Green, Pink, Purple, Orange
misclassification_color = 'black'

# Draw lines for each sample, color-coded by class
for i in range(n_samples):
    color_index = true_labels[i]  # No adjustment needed, as labels are 0-4
    color = class_colors[color_index]  # Directly use true_labels[i] as the index for colors
    ax.hlines(y=y_positions[i], xmin=0, xmax=bar_width, colors=color, lw=1.0)

# Overlay misclassifications as black lines
for i in range(n_samples):
    if true_labels[i] != predicted_labels[i]:  # Misclassified instance
        ax.hlines(y=y_positions[i], xmin=0, xmax=bar_width, colors=misclassification_color, lw=1.0)

# Add horizontal lines at each class boundary
unique_labels, class_counts = np.unique(true_labels, return_counts=True)
class_boundaries = np.cumsum(class_counts)

for boundary in class_boundaries[:-1]:  # Skip the last boundary as it would go beyond the plot
    ax.axhline(y=boundary - 0.5, color='black', linestyle='--', lw=0.8)

# Customize plot labels, ticks, and legend
ax.set_title('Ensemble Model Predictions (Grouped by Classes)')
ax.set_xlabel('Ensemble Model')
ax.set_ylabel('Index')
ax.set_xticks([])  # No x-axis ticks since we only have one ensemble model
ax.set_yticks([])

# Create a legend with colors for each class and misclassification marker
legend_labels = [f'Class {i}' for i in range(len(class_colors))] + ['Misclassified']
legend_handles = [plt.Line2D([0], [0], color=color, lw=2) for color in class_colors] + \
                 [plt.Line2D([0], [0], color=misclassification_color, lw=2)]
ax.legend(legend_handles, legend_labels, loc='lower right', title="Legend")

plt.tight_layout()
plt.savefig('ensemble_model_strip_plot.png')
plt.show()
print("Plot saved as 'ensemble_model_strip_plot.png'")
