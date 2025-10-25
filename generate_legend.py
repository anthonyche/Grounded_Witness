"""
Generate a standalone legend figure for MUTAG visualizations.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# MUTAG atom types and colors (same as in visualize_case_study.py)
atom_names = ['C', 'N', 'O', 'F', 'I', 'Cl', 'Br']
atom_colors = ['#F5DEB3', '#87CEEB', '#FFB6C1', '#98FB98', '#DDA0DD', '#90EE90', '#6A0DAD']
# C: Wheat, N: SkyBlue, O: LightPink, F: PaleGreen, I: Plum, Cl: LightGreen, Br: DarkPurple

# Create figure - square shape for LaTeX
fig, ax = plt.subplots(figsize=(1.1, 1.1))
ax.axis('off')

# Build legend elements
legend_elements = []

# Atom type legend - with thinner edges
for i in range(len(atom_names)):
    legend_elements.append(
        Patch(facecolor=atom_colors[i], edgecolor='black', linewidth=0.5, label=atom_names[i])
    )

# Add separator
legend_elements.append(Line2D([0], [0], color='none', label=''))

# Edge type legend

legend_elements.append(
    Line2D([0], [0], color='red', linewidth=1.2, label='Witness edges')
)
legend_elements.append(
    Line2D([0], [0], color='royalblue', linewidth=1.6, linestyle='dotted', label='Grounded edges')
)
legend_elements.append(
    Line2D([0], [0], color="#000000", linewidth=0.6, label='Original edges')
)


# Create legend - square frame, no title, longer handles
legend = ax.legend(
    handles=legend_elements,
    loc='center',
    framealpha=0.95,
    fontsize=4.8,
    ncol=1,
    edgecolor='black',
    fancybox=False,
    handlelength=1.4,
    handleheight=0.4,
    labelspacing=0.2,
    borderpad=0.32,
    handletextpad=0.4
)

plt.tight_layout(pad=0)
plt.savefig('mutag_legend.png', dpi=200, bbox_inches='tight', pad_inches=0.02)
print("Legend saved to: mutag_legend.png")
plt.close()
