import matplotlib.pyplot as plt
import numpy as np

# --- Data Definition ---
# X-axis values
x_full = [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
x_pt3r_scan = [50, 100, 150, 200, 300, 400, 500, 600, 700]
x_str_vggt = [50, 100, 150]
x_pt3r_tum = [50, 100, 150, 200, 300, 400, 500, 600]

# Scannet (Left Plot) Approximate Data
scannet_vggt = [0.02, 0.03, 0.04]
scannet_str_vggt = [0.070, 0.185, 0.325]
scannet_cut3r = [0.04, 0.11, 0.21, 0.32, 0.47, 0.57, 0.66, 0.71, 0.73, 0.76, 0.79, 0.82]
scannet_pt3r = [0.070, 0.140, 0.240, 0.290, 0.450, 0.510, 0.530, 0.590, 0.635]
scannet_ttt3r = [0.033, 0.072, 0.11, 0.14, 0.19, 0.24, 0.28, 0.32, 0.344, 0.37, 0.40, 0.40]
scannet_ttsa3r = [0.033, 0.063, 0.10, 0.13, 0.17, 0.21, 0.26, 0.30, 0.33, 0.36, 0.38, 0.39]
scannet_ivggt = [0.030, 0.060, 0.080, 0.120, 0.200, 0.265, 0.330, 0.355, 0.400, 0.44, 0.47, 0.48]
scannet_cut3r_s = [0.040, 0.083, 0.14, 0.20, 0.33, 0.43, 0.53, 0.58, 0.62, 0.65, 0.67, 0.71]
scannet_cut3r_s_ttt3r = [0.034, 0.068, 0.10, 0.13, 0.17, 0.20, 0.22, 0.23, 0.25, 0.27, 0.28, 0.29]
scannet_cut3r_s_ttsa3r = [0.035, 0.065, 0.09, 0.12, 0.16, 0.18, 0.21, 0.23, 0.24, 0.26, 0.28, 0.30]


# TUM Dynamics (Right Plot) Approximate Data
tum_vggt = [0.005, 0.008, 0.009]
tum_str_vggt = [0.010, 0.025, 0.038]
tum_cut3r = [0.015, 0.033, 0.042, 0.061, 0.086, 0.123, 0.141, 0.150, 0.168, 0.171, 0.172, 0.173]
tum_pt3r = [0.024, 0.042, 0.076, 0.079, 0.106, 0.149, 0.130, 0.166]
tum_ttt3r = [0.014, 0.02, 0.031, 0.041, 0.045, 0.051, 0.067, 0.074, 0.085, 0.091, 0.102, 0.11]
tum_ttsa3r = [0.012, 0.018, 0.025, 0.030, 0.038, 0.043, 0.056, 0.064, 0.074, 0.079, 0.086, 0.091]
tum_ivggt = [0.013, 0.019, 0.026, 0.031, 0.035, 0.038, 0.047, 0.052, 0.058, 0.062, 0.065, 0.066]
tum_cut3r_s = [0.012, 0.020, 0.031, 0.042, 0.064, 0.078, 0.096, 0.11, 0.13, 0.14, 0.15, 0.15]
tum_cut3r_s_ttt3r = [0.011, 0.015, 0.019, 0.025, 0.031, 0.037, 0.041, 0.046, 0.053, 0.060, 0.066, 0.073]
tum_cut3r_s_ttsa3r = [0.011, 0.015, 0.018, 0.023, 0.027, 0.032, 0.036, 0.039, 0.045, 0.051, 0.058, 0.064]


# Formatting configurations
styles = {
    # --- Independent Methods ---
    'VGGT_offline':    {'color': 'tab:purple', 'marker': '*', 'linestyle': '-'},
    'InfiniteVGGT':    {'color': 'tab:orange', 'marker': 'X', 'linestyle': '-'},
    'StreamVGGT':      {'color': 'tab:red',   'marker': 'o', 'linestyle': '-'},
    'Point3R':         {'color': 'sienna',     'marker': 'P', 'linestyle': '-'},

    # --- Base Methods ---
    'CUT3R':           {'color': 'tab:blue',   'marker': 's', 'linestyle': '-'},
    'TTT3R':           {'color': 'tab:gray',   'marker': 'p', 'linestyle': '-'},
    'TTSA3R':          {'color': 'tab:green',  'marker': 'D', 'linestyle': '-'},

    # --- Fly3R Method Group ---
    'Ours':           {'color': 'tab:blue', 'linestyle': '--'},
    'Ours + TTT3R':  {'color': 'tab:gray', 'linestyle': '--'},
    'Ours + TTSA3R': {'color': 'tab:green', 'linestyle': '--'},
}

def style_axis(ax, title):
    ax.set_title(title, fontsize=16, pad=10)
    ax.set_xlabel('Number of Input Views', fontsize=14, labelpad=8)
    ax.set_ylabel(r'ATE (m) $\downarrow$', fontsize=14, labelpad=8)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=13)
    for spine in ax.spines.values():
        spine.set_color('lightgrey')
        spine.set_linewidth(1.5)

# ==========================================
# 1. Save as a single combined PDF (1x2)
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), dpi=150)
plt.subplots_adjust(wspace=0.25)

# Plot Scannet
style_axis(ax1, 'Pose Estimation (ATE) - ScanNet')
ax1.plot(x_str_vggt, scannet_vggt, label='VGGT_offline', lw=2.5, **styles['VGGT_offline'])
ax1.plot(x_str_vggt, scannet_str_vggt, label='StreamVGGT', lw=2.5, **styles['StreamVGGT'])
ax1.plot(x_pt3r_scan, scannet_pt3r, label='Point3R', lw=2.5, **styles['Point3R'])
ax1.plot(x_full, scannet_ivggt, label='InfiniteVGGT', lw=2.5, **styles['InfiniteVGGT'])
ax1.plot(x_full, scannet_cut3r, label='CUT3R', lw=2.5, **styles['CUT3R'])
ax1.plot(x_full, scannet_ttt3r, label='TTT3R', lw=2.5, **styles['TTT3R'])
ax1.plot(x_full, scannet_ttsa3r, label='TTSA3R', lw=2.5, **styles['TTSA3R'])
ax1.plot(x_full, scannet_cut3r_s, label='Ours', lw=2.5, **styles['Ours'])
ax1.plot(x_full, scannet_cut3r_s_ttt3r, label='Ours + TTT3R', lw=2.5, **styles['Ours + TTT3R'])
ax1.plot(x_full, scannet_cut3r_s_ttsa3r, label='Ours + TTSA3R', lw=2.5, **styles['Ours + TTSA3R'])
ax1.legend(loc='upper left', shadow=False, fancybox=True, framealpha=0.9)

# Plot TUM Dynamics
style_axis(ax2, 'Pose Estimation (ATE) - TUM Dynamics')
ax2.plot(x_str_vggt, tum_vggt, lw=2.5, **styles['VGGT_offline'])
ax2.plot(x_str_vggt, tum_str_vggt, lw=2.5, **styles['StreamVGGT'])
ax2.plot(x_pt3r_tum, tum_pt3r, lw=2.5, **styles['Point3R'])
ax2.plot(x_full, tum_ivggt, lw=2.5, **styles['InfiniteVGGT'])
ax2.plot(x_full, tum_cut3r, lw=2.5, **styles['CUT3R'])
ax2.plot(x_full, tum_ttt3r, lw=2.5, **styles['TTT3R'])
ax2.plot(x_full, tum_ttsa3r, lw=2.5, **styles['TTSA3R'])
ax2.plot(x_full, tum_cut3r_s, lw=2.5, **styles['Ours'])
ax2.plot(x_full, tum_cut3r_s_ttt3r, lw=2.5, **styles['Ours + TTT3R'])
ax2.plot(x_full, tum_cut3r_s_ttsa3r, lw=2.5, **styles['Ours + TTSA3R'])

fig.savefig('pose_estimation_combined.pdf', bbox_inches='tight')
plt.close(fig)

# ==========================================
# 2. Save Scannet as a separate PDF
# ==========================================
fig1, ax1_ind = plt.subplots(figsize=(8, 5), dpi=150)
style_axis(ax1_ind, 'Pose Estimation (ATE) - scannet')
ax1_ind.plot(x_str_vggt, scannet_vggt, label='VGGT_offline', lw=2.5, **styles['VGGT_offline'])
ax1_ind.plot(x_str_vggt, scannet_str_vggt, label='StreamVGGT', lw=2.5, **styles['StreamVGGT'])
ax1_ind.plot(x_pt3r_scan, scannet_pt3r, label='Point3R', lw=2.5, **styles['Point3R'])
ax1_ind.plot(x_full, scannet_cut3r, label='CUT3R', lw=2.5, **styles['CUT3R'])
ax1_ind.plot(x_full, scannet_ttt3r, label='TTT3R', lw=2.5, **styles['TTT3R'])
ax1_ind.plot(x_full, scannet_ttsa3r, label='TTSA3R', lw=2.5, **styles['TTSA3R'])
ax1_ind.plot(x_full, scannet_cut3r_s, label='Ours', lw=2.5, **styles['Ours'])
ax1_ind.plot(x_full, scannet_cut3r_s_ttt3r, label='Ours + TTT3R', lw=2.5, **styles['Ours + TTT3R'])
ax1_ind.plot(x_full, scannet_cut3r_s_ttsa3r, label='Ours + TTSA3R', lw=2.5, **styles['Ours + TTSA3R'])
ax1_ind.legend(loc='upper left', shadow=False, fancybox=True, framealpha=0.9)

fig1.savefig('pose_estimation_scannet.pdf', bbox_inches='tight')
plt.close(fig1)

# ==========================================
# 3. Save TUM Dynamics as a separate PDF
# ==========================================
fig2, ax2_ind = plt.subplots(figsize=(8, 5), dpi=150)
style_axis(ax2_ind, 'Pose Estimation (ATE) - tum dynamics')
ax2_ind.plot(x_str_vggt, tum_vggt, lw=2.5, **styles['VGGT_offline'])
ax2_ind.plot(x_str_vggt, tum_str_vggt, lw=2.5, **styles['StreamVGGT'])
ax2_ind.plot(x_pt3r_tum, tum_pt3r, lw=2.5, **styles['Point3R'])
ax2_ind.plot(x_full, tum_cut3r, lw=2.5, **styles['CUT3R'])
ax2_ind.plot(x_full, tum_ttt3r, lw=2.5, **styles['TTT3R'])
ax2_ind.plot(x_full, tum_ttsa3r, lw=2.5, **styles['TTSA3R'])
ax2_ind.plot(x_full, tum_cut3r_s, lw=2.5, **styles['Ours'])
ax2_ind.plot(x_full, tum_cut3r_s_ttt3r, lw=2.5, **styles['Ours + TTT3R'])
ax2_ind.plot(x_full, tum_cut3r_s_ttsa3r, lw=2.5, **styles['Ours + TTSA3R'])

fig2.savefig('pose_estimation_tum.pdf', bbox_inches='tight')
plt.close(fig2)

print("All PDFs successfully generated!")
