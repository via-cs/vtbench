import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("updated_dataset.xlsx")

# Create two separate dataframes based on Class column
df_binary = df[df["Class"] == 2].copy()
df_multiclass = df[df["Class"] > 2].copy()

# Sort both dataframes by ESI_CKA in descending order
df_binary = df_binary.sort_values(by="ESI_CKA", ascending=False)
df_multiclass = df_multiclass.sort_values(by="ESI_CKA", ascending=False)

# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Plot for binary class datasets
if not df_binary.empty:
    bar_width = 0.6  # Thinner bars
    x_pos = range(len(df_binary))
    
    ax1.bar(x_pos, df_binary["ESI_CKA"], width=bar_width, label="ESI_CKA")
    ax1.bar(x_pos, df_binary["ESI_PROBE"], width=bar_width, bottom=df_binary["ESI_CKA"], label="ESI_PROBE")
    
    ax1.set_ylabel("Score", fontsize=14)
    ax1.set_title("Binary Class Datasets", fontsize=16, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(df_binary["Dataset"], rotation=45, ha="right", fontsize=11)
    ax1.legend(fontsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    
    # Remove white space from plot area
    ax1.set_xlim(-0.5, len(df_binary) - 0.5)
    ax1.margins(x=0)
else:
    ax1.text(0.5, 0.5, 'No binary class datasets', 
             horizontalalignment='center', verticalalignment='center',
             transform=ax1.transAxes, fontsize=14)
    ax1.set_title("Binary Class Datasets", fontsize=16, fontweight='bold')

# Plot for multiclass datasets
if not df_multiclass.empty:
    bar_width = 0.6  # Thinner bars
    x_pos = range(len(df_multiclass))
    
    ax2.bar(x_pos, df_multiclass["ESI_CKA"], width=bar_width, label="ESI_CKA")
    ax2.bar(x_pos, df_multiclass["ESI_PROBE"], width=bar_width, bottom=df_multiclass["ESI_CKA"], label="ESI_PROBE")
    
    ax2.set_ylabel("Score", fontsize=14)
    ax2.set_title("Multiclass Datasets", fontsize=16, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(df_multiclass["Dataset"], rotation=45, ha="right", fontsize=11)
    ax2.legend(fontsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    
    # Remove white space from plot area
    ax2.set_xlim(-0.5, len(df_multiclass) - 0.5)
    ax2.margins(x=0)
else:
    ax2.text(0.5, 0.5, 'No multiclass datasets', 
             horizontalalignment='center', verticalalignment='center',
             transform=ax2.transAxes, fontsize=14)
    ax2.set_title("Multiclass Datasets", fontsize=16, fontweight='bold')

# Overall figure title
fig.suptitle("ESI Comparison Across Datasets", fontsize=18, fontweight='bold', y=1.02)

# Adjust layout to remove white space
plt.tight_layout()
plt.savefig("stacked_esi_plot_split.png", dpi=300, bbox_inches='tight')
plt.show()