import matplotlib.pyplot as plt

# Apply a serif font family to match the academic aesthetic
plt.rcParams['font.family'] = 'sans-serif'

# ---------------------------------------------------------
# 1. Data Setup (Original + Placeholders for New Methods)
# ---------------------------------------------------------
# X-axis values for full evaluation and methods that ran out of memory (OOM)
x_full = [50, 100, 150, 200, 250, 300, 350, 400]
x_oom = [50, 100]

# --- Left Plot (Chamfer Distance) Data ---
cd_vg   = [0.019, 0.018]
cd_sv   = [0.023, 0.028]
cd_pt   = [0.020, 0.027, 0.029, 0.038, 0.039, 0.042, 0.043, 0.050]
cd_cut  = [0.019, 0.028, 0.044, 0.070, 0.096, 0.110, 0.118, 0.133]
cd_ttt  = [0.018, 0.018, 0.021, 0.025, 0.030, 0.031, 0.034, 0.038]

# Placeholders for new methods (Replace with your actual data)
cd_ttsa = [0.019, 0.018, 0.020, 0.023, 0.025, 0.026, 0.027, 0.030]
cd_ours = [0.020, 0.020, 0.027, 0.038, 0.052, 0.057, 0.066, 0.077]
cd_ours_ttt = [0.019, 0.018, 0.020, 0.023, 0.025, 0.025, 0.025, 0.027]
cd_ours_ttsa= [0.019, 0.019, 0.019, 0.021, 0.022, 0.022, 0.023, 0.023]

# --- Right Plot (Normal Consistency) Data ---
nc_vg   = [0.627, 0.605]
nc_sv   = [0.615, 0.596]
nc_pt   = [0.610, 0.600, 0.590, 0.585, 0.578, 0.570, 0.564, 0.560]
nc_cut  = [0.607, 0.591, 0.577, 0.563, 0.550, 0.541, 0.536, 0.533]
nc_ttt  = [0.610, 0.594, 0.587, 0.581, 0.572, 0.565, 0.560, 0.557]

# Placeholders for new methods (Replace with your actual data)
nc_ttsa = [0.610, 0.594, 0.588, 0.582, 0.573, 0.567, 0.563, 0.561]
nc_ours = [0.610, 0.592, 0.584, 0.575, 0.565, 0.558, 0.554, 0.549]
nc_ours_ttt = [0.609, 0.592, 0.587, 0.580, 0.572, 0.566, 0.563, 0.560]
nc_ours_ttsa= [0.608, 0.592, 0.587, 0.581, 0.572, 0.566, 0.563, 0.560]

# ---------------------------------------------------------
# 2. Styling (Colors, Markers, Linewidths)
# ---------------------------------------------------------
colors = {
    'VGGT': '#9467bd',       # Purple
    'StreamVGGT': '#d62728', # Red
    'Point3R': '#8c564b',    # Brown
    'CUT3R': '#1f77b4',      # Blue
    'TTT3R': '#7f7f7f',      # Gray
    'TTSA3R': '#2ca02c',     # Green
    'Ours': '#1f77b4',       # Blue
    'Ours+TTT3R': '#7f7f7f', # Gray
    'Ours+TTSA3R': '#2ca02c' # Green
}

markers = {
    'VGGT': '*',       # Star
    'StreamVGGT': 'o', # Circle
    'Point3R': 'P',    # Filled Plus / Cross
    'CUT3R': 's',      # Square
    'TTT3R': 'p',      # Pentagon
    'TTSA3R': 'D'      # Diamond
}

lw = 2.0 # Linewidth
ms = 6   # Markersize

# ---------------------------------------------------------
# 3. Plot Setup and Plotting Function
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 3.5))

def plot_lines(ax, d_vg, d_sv, d_pt, d_cut, d_ttt, d_ttsa, d_ours, d_ours_ttt, d_ours_ttsa, is_cd):
    # 1. Baselines (Solid lines with defined markers)
    ax.plot(x_oom, d_vg, marker=markers['VGGT'], label='VGGT (offline)', color=colors['VGGT'], linewidth=lw, markersize=ms)
    ax.plot(x_oom, d_sv, marker=markers['StreamVGGT'], label='StreamVGGT', color=colors['StreamVGGT'], linewidth=lw, markersize=ms)
    ax.plot(x_full, d_pt, marker=markers['Point3R'], label='Point3R', color=colors['Point3R'], linewidth=lw, markersize=ms)
    ax.plot(x_full, d_cut, marker=markers['CUT3R'], label='CUT3R', color=colors['CUT3R'], linewidth=lw, markersize=ms)
    ax.plot(x_full, d_ttt, marker=markers['TTT3R'], label='TTT3R', color=colors['TTT3R'], linewidth=lw, markersize=ms)
    ax.plot(x_full, d_ttsa, marker=markers['TTSA3R'], label='TTSA3R', color=colors['TTSA3R'], linewidth=lw, markersize=ms)

    # 2. Ours variants (Dashed lines, no markers to distinguish them visually)
    ax.plot(x_full, d_ours, linestyle='--', marker='None', label='Ours', color=colors['Ours'], linewidth=lw)
    ax.plot(x_full, d_ours_ttt, linestyle='--', marker='None', label='Ours + TTT3R', color=colors['Ours+TTT3R'], linewidth=lw)
    ax.plot(x_full, d_ours_ttsa, linestyle='--', marker='None', label='Ours + TTSA3R', color=colors['Ours+TTSA3R'], linewidth=lw)

    # 3. Add "OOM" (Out of Memory) annotations next to the final points of VGGT and StreamVGGT
    # We add +5 to the x-coordinate to place the text slightly to the right of the marker
    # va='center' ensures the text vertically aligns directly with the point
    ax.text(x_oom[-1] + 5, d_sv[-1], 'OOM', color=colors['StreamVGGT'], fontweight='bold', fontsize=10, va='center')
    ax.text(x_oom[-1] + 5, d_vg[-1], 'OOM', color=colors['VGGT'], fontweight='bold', fontsize=10, va='center')

    # Grid and Tick styling
    ax.set_xticks(x_full)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='#CCCCCC')
    ax.set_xlabel('Number of Input Views', fontsize=12)

    # Make the plot spines (borders) lighter to match the reference style
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('#DDDDDD')

# ---------------------------------------------------------
# 4. Generate Subplots & Save
# ---------------------------------------------------------
# Left Plot: Chamfer Distance
plot_lines(axes[0], cd_vg, cd_sv, cd_pt, cd_cut, cd_ttt, cd_ttsa, cd_ours, cd_ours_ttt, cd_ours_ttsa, is_cd=True)
# Use 'r' before the string to treat it as a raw string, allowing LaTeX commands like \downarrow
axes[0].set_ylabel(r'Chamfer Distance $\downarrow$', fontsize=12)
axes[0].legend(loc='upper left', fontsize=9, framealpha=1)

# Right Plot: Normal Consistency
plot_lines(axes[1], nc_vg, nc_sv, nc_pt, nc_cut, nc_ttt, nc_ttsa, nc_ours, nc_ours_ttt, nc_ours_ttsa, is_cd=False)
axes[1].set_ylabel(r'Normal Consistency $\uparrow$', fontsize=12)

# Adjust layout to prevent overlaps
plt.tight_layout()

# Save the figure as a PDF (bbox_inches='tight' prevents labels from being cut off)
plt.savefig('3d_recon_7s.pdf', format='pdf', bbox_inches='tight')
