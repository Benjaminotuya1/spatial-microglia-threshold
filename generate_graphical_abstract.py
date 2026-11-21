import os
import warnings
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
})

DATA_DIR = os.path.join('Reanalysis_2026', 'data')
FIG_DIR = os.path.join('Reanalysis_2026', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# 1. Compute real CosMx CP1K Cd74+Ctss curve (0-60 um) for embedded mini-plot in Column 3
print("Loading CosMx data for real embedded mini-plot...", flush=True)
dfs_c = []
for gsm in ['GSM8199188', 'GSM8199189']:
    meta = pd.read_csv(os.path.join(DATA_DIR, f'{gsm}_metadata.csv.gz'))
    expr_c = pd.read_csv(os.path.join(DATA_DIR, f'{gsm}_exprMat.csv.gz'))
    gene_cols_all = [c for c in expr_c.columns if c not in ['fov', 'cell_ID'] and not c.startswith('Negative') and not c.startswith('SystemControl')]
    tot_tx = expr_c[gene_cols_all].sum(axis=1).values
    keep_cols = ['fov', 'cell_ID', 'Cd74', 'Ctss', 'C1qa', 'C1qb', 'C1qc', 'Hexb', 'Csf1r', 'Tyrobp']
    sub_e = expr_c[keep_cols].copy()
    sub_e['Total_Transcripts'] = tot_tx
    m = pd.merge(meta, sub_e, on=['fov', 'cell_ID'])
    px_to_um = 0.12028
    m['X_um'] = m['CenterX_global_px'] * px_to_um
    m['Y_um'] = m['CenterY_global_px'] * px_to_um

    abeta_thresh = m['Mean.BetaAmyloid'].quantile(0.98)
    high_abeta_mask = (m['Mean.BetaAmyloid'] >= abeta_thresh)
    high_abeta_xy = m.loc[high_abeta_mask, ['X_um', 'Y_um']].values
    t_ab = cKDTree(high_abeta_xy)
    adj_ab = t_ab.sparse_distance_matrix(t_ab, max_distance=25.0)
    _, plaque_labels = connected_components(adj_ab, directed=False)

    high_abeta_df = m.loc[high_abeta_mask, ['X_um', 'Y_um', 'Mean.BetaAmyloid']].copy()
    high_abeta_df['Plaque_ID'] = plaque_labels
    p_cents = []
    for pid, pgrp in high_abeta_df.groupby('Plaque_ID'):
        w = pgrp['Mean.BetaAmyloid'].values
        p_cents.append({'X_um': np.sum(pgrp['X_um'].values * w) / np.sum(w), 'Y_um': np.sum(pgrp['Y_um'].values * w) / np.sum(w)})
    p_df = pd.DataFrame(p_cents)

    m['MG_Score'] = m['C1qa'] + m['C1qb'] + m['C1qc'] + m['Hexb'] + m['Csf1r'] + m['Tyrobp']
    mg_c = m[m['MG_Score'] >= 4].copy()
    t_cent = cKDTree(p_df[['X_um', 'Y_um']].values)
    d_cent, _ = t_cent.query(mg_c[['X_um', 'Y_um']].values, k=2)
    mg_c['Dist_Plaque_um'] = np.where(d_cent[:, 0] == 0, d_cent[:, 1], d_cent[:, 0])
    tot = np.maximum(mg_c['Total_Transcripts'].values, 1.0)
    mg_c['Norm_CD74_CTSS_CP1K'] = 1000.0 * (mg_c['Cd74'] + mg_c['Ctss']) / tot
    dfs_c.append(mg_c)

cosmx_all = pd.concat(dfs_c, ignore_index=True)
c_140 = cosmx_all[(cosmx_all['Dist_Plaque_um'] > 0) & (cosmx_all['Dist_Plaque_um'] <= 140.0)].copy()
bins_c = np.arange(0, 145, 5)
c_140['Bin'] = pd.cut(c_140['Dist_Plaque_um'], bins=bins_c, labels=bins_c[:-1] + 2.5, include_lowest=True).astype(float)
prof_c = c_140.groupby('Bin', observed=False).agg(
    norm_cd74=('Norm_CD74_CTSS_CP1K', 'mean'),
    se_cd74=('Norm_CD74_CTSS_CP1K', lambda s: s.std() / np.sqrt(len(s))),
).dropna()

bx_all = prof_c.index.values
X_all = np.column_stack([np.ones_like(bx_all), np.minimum(bx_all, 33), np.maximum(0, bx_all - 33)])
fit_all = sm.OLS(prof_c['norm_cd74'].values, X_all).fit()
prof_60 = prof_c[prof_c.index <= 60.0]
bx_60 = prof_60.index.values
X_60 = np.column_stack([np.ones_like(bx_60), np.minimum(bx_60, 33), np.maximum(0, bx_60 - 33)])
pred_60 = fit_all.predict(X_60)

# 2. Build 16:9 High-Resolution Vector-Grade Graphical Abstract (16 x 9 inches, 300 DPI)
print("Rendering Graphical Abstract...", flush=True)
fig = plt.figure(figsize=(16.0, 9.0), dpi=300, facecolor='#F4F6F9')
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

def add_card(x, y, w, h, fc='white', ec='#CCCCCC', lw=1.4, r=0.18, zorder=1):
    box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={r}",
                         facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    return box

# --- TOP HEADER BANNER ---
add_card(0.25, 7.82, 15.50, 0.96, fc='#1B365D', ec='#1B365D', lw=0, r=0.15, zorder=2)
ax.text(8.0, 8.46, "Methodological Audit & Direct Physical Validation of Peri-Plaque Microglial Activation in Alzheimer's Disease",
        ha='center', va='center', fontsize=15.8, fontweight='bold', color='white', zorder=3)
ax.text(8.0, 8.06, "1,887,729 SEA-AD Human MTG MERFISH Cells (27 Donors, 69 Sections)  +  402 Physical Anti-Aβ Plaques in CosMx (GSE263791)",
        ha='center', va='center', fontsize=11.0, color='#D6EAF8', zorder=3)

# --- 3 MAIN COLUMN CARDS ---
col_y = 1.05
col_h = 6.50
col_w = 4.86
c1_x, c2_x, c3_x = 0.28, 5.57, 10.86

# Column 1: 5 Methodological Confounders
add_card(c1_x, col_y, col_w, col_h, fc='white', ec='#C0392B', lw=1.8, r=0.16, zorder=1)
add_card(c1_x, col_y + col_h - 0.68, col_w, 0.68, fc='#C0392B', ec='#C0392B', lw=0, r=0.14, zorder=2)
ax.text(c1_x + col_w/2, col_y + col_h - 0.34, "1. FIVE CONFOUNDERS EXPOSED (SEA-AD)",
        ha='center', va='center', fontsize=11.6, fontweight='bold', color='white', zorder=3)

confounders = [
    ("(1) Pooled-Section Coordinate Artifact",
     "Overlaying 69 sections pulls apparent knots to ~24 µm.\nStrictly within-section proxy knots scatter across 33–71 µm\n(Median 56 µm; power collapses to 0–22.5% in thin slices).",
     '#FDEDEC', '#E6B0AA'),
    ("(2) Two-Tier Neighbor Transcript Spillover",
     "Tier-1 (Broad Class): 83/138 genes flagged ≥1.5× (RBFOX3 2.48×).\nTier-2 (Rare Subclass): TACR1 (2.32× Sst Chodl), PLD5 (2.17×),\nTMEM255A (1.75× OPC), RFX3 (1.60× Endothelial).",
     '#FEF5E7', '#F5CBA7'),
    ("(3) Single-Donor Pseudoreplication",
     "One mega-donor (H20.33.004) supplies 65.5% of Dementia &\n82.0% of APOE4+ proximal cells. Donor-level MWU tests\nare NS (Dementia p = 0.114; APOE4 p = 0.289).",
     '#FDEDEC', '#E6B0AA'),
    ("(4) Matrix Metadata & (5) Capture-Rate Bias",
     "SEA-AD adata.X stores volume-scaled linear counts (max 483),\nnot log1p. Unnormalized CosMx counts distort homeostatic\nmarkers (CP1K normalization corrects the artifact).",
     '#F4F6F7', '#BDC3C7'),
]

cy = col_y + col_h - 0.90
for title_c, body_c, bg_c, bd_c in confounders:
    bh = 1.24
    add_card(c1_x + 0.18, cy - bh, col_w - 0.36, bh, fc=bg_c, ec=bd_c, lw=1.1, r=0.10, zorder=2)
    ax.text(c1_x + 0.30, cy - 0.22, title_c, ha='left', va='center', fontsize=9.8, fontweight='bold', color='#1C2833', zorder=3)
    ax.text(c1_x + 0.30, cy - 0.74, body_c, ha='left', va='center', fontsize=8.35, color='#2C3E50', linespacing=1.35, zorder=3)
    cy -= (bh + 0.16)

# Column 2: Spillover-Free ML & Surviving Biology
add_card(c2_x, col_y, col_w, col_h, fc='white', ec='#117A65', lw=1.8, r=0.16, zorder=1)
add_card(c2_x, col_y + col_h - 0.68, col_w, 0.68, fc='#117A65', ec='#117A65', lw=0, r=0.14, zorder=2)
ax.text(c2_x + col_w/2, col_y + col_h - 0.34, "2. SPILLOVER-FREE ML & INTRINSIC CORE",
        ha='center', va='center', fontsize=11.6, fontweight='bold', color='white', zorder=3)

# Embedded Mini-Plot 1: Donor-CV Feature Ablation Bars (inside top half of Col 2)
ax_m2 = fig.add_axes([0.382, 0.515, 0.245, 0.225])
ab_lbls = ['Genes+Dist\n(139)', 'Genes Only\n(138)', 'Decontam\n(54)', 'Dist Only\n(WithinSec)']
ab_vals = [0.8917, 0.8912, 0.8810, 0.5568]
ab_cols = ['#117A65', '#2E86C1', '#16A085', '#95A5A6']
b_m2 = ax_m2.bar(range(4), ab_vals, color=ab_cols, edgecolor='#1B365D', width=0.58)
for b, v in zip(b_m2, ab_vals):
    ax_m2.text(b.get_x() + b.get_width()/2, v + 0.018, f"{v:.4f}", ha='center', va='bottom', fontsize=8.0, fontweight='bold', color='#1C2833')
ax_m2.set_xticks(range(4))
ax_m2.set_xticklabels(ab_lbls, fontsize=7.6, fontweight='bold')
ax_m2.set_ylim(0.45, 1.02)
ax_m2.set_ylabel("5-Fold Donor ROC-AUC", fontsize=8.2, fontweight='bold')
ax_m2.set_title("Donor-Level Generalization (27 Donors; Median = 0.8821)", fontsize=8.8, fontweight='bold', color='#1B365D', pad=5)
ax_m2.tick_params(axis='y', labelsize=7.5)
ax_m2.grid(True, alpha=0.25, axis='y')
ax_m2.spines['top'].set_visible(False)
ax_m2.spines['right'].set_visible(False)

# Intrinsic Myeloid Core Box
add_card(c2_x + 0.18, 2.75, col_w - 0.36, 1.45, fc='#E8F8F5', ec='#117A65', lw=1.2, r=0.10, zorder=2)
ax.text(c2_x + col_w/2, 3.96, "4 Spillover-Resistant Intrinsic Myeloid Drivers", ha='center', va='center', fontsize=9.8, fontweight='bold', color='#0E6251', zorder=3)
ax.text(c2_x + col_w/2, 3.65, "(All pass <1.5× spillover across all 24 SEA-AD subclasses)", ha='center', va='center', fontsize=8.1, color='#1E8449', zorder=3)
drivers = [('LRRK1', '0.87×'), ('L3MBTL4', '1.41×'), ('PDE4B', '1.42×'), ('UNC5B', '1.32×')]
for idx_d, (dg, dr) in enumerate(drivers):
    dx = c2_x + 0.30 + idx_d * 1.11
    add_card(dx, 2.90, 0.98, 0.56, fc='#117A65', ec='#0B5345', lw=1.0, r=0.08, zorder=3)
    ax.text(dx + 0.49, 3.27, dg, ha='center', va='center', fontsize=8.8, fontweight='bold', color='white', zorder=4)
    ax.text(dx + 0.49, 3.04, f"max {dr}", ha='center', va='center', fontsize=7.4, color='#D1F2EB', zorder=4)

# Surviving Within-Section Biology Box
add_card(c2_x + 0.18, 1.24, col_w - 0.36, 1.32, fc='#EBF5FB', ec='#2980B9', lw=1.2, r=0.10, zorder=2)
ax.text(c2_x + 0.30, 2.34, "Surviving Within-Section Human Biology (X_spatial_tiled)", ha='left', va='center', fontsize=9.4, fontweight='bold', color='#1B4F72', zorder=3)
surv_txt = (
    "• Continuous Disease Progression (CPS vs. Donor ΔScore):\n"
    "   Spearman ρ = +0.397, p = 0.0404 across all 27 donors\n"
    "• Multicellular Niche Remodeling (<25 µm vs. Distal):\n"
    "   −39.4% Neuronal Depletion  &  +58.2% Microglial Enrichment"
)
ax.text(c2_x + 0.30, 1.72, surv_txt, ha='left', va='center', fontsize=8.25, color='#1C2833', linespacing=1.38, zorder=3)

# Column 3: Direct Physical Plaque Validation (CosMx)
add_card(c3_x, col_y, col_w, col_h, fc='white', ec='#D35400', lw=1.8, r=0.16, zorder=1)
add_card(c3_x, col_y + col_h - 0.68, col_w, 0.68, fc='#D35400', ec='#D35400', lw=0, r=0.14, zorder=2)
ax.text(c3_x + col_w/2, col_y + col_h - 0.34, "3. DIRECT Aβ PLAQUE VALIDATION (CosMx)",
        ha='center', va='center', fontsize=11.6, fontweight='bold', color='white', zorder=3)

# Embedded Real CosMx Mini-Plot (inside top half of Col 3)
ax_m3 = fig.add_axes([0.713, 0.515, 0.245, 0.225])
ax_m3.errorbar(bx_60, prof_60['norm_cd74'].values, yerr=prof_60['se_cd74'].values, fmt='o-', color='#C0392B',
               ecolor='#E6B0AA', lw=2.0, markersize=4.2, capsize=2, label='Cd74+Ctss (CP1K)')
ax_m3.plot(bx_60, pred_60, 'k--', lw=1.8, label='Hinge Fit (33 µm)')
ax_m3.axvline(33.0, color='#117A65', ls=':', lw=2.0)
ax_m3.axvspan(31.0, 37.0, color='#117A65', alpha=0.18, label='95% CI (31–37 µm)')
ax_m3.set_xlim(0, 62)
ax_m3.set_xlabel("Distance to True Aβ Plaque Centroid (µm)", fontsize=7.8, fontweight='bold')
ax_m3.set_ylabel("CP1K ± SE", fontsize=8.0, fontweight='bold')
ax_m3.set_title("402 Anti-Aβ Plaques (MOAB-2 IF, d=0 Excl.)", fontsize=8.8, fontweight='bold', color='#922B21', pad=5)
ax_m3.tick_params(labelsize=7.4)
ax_m3.legend(fontsize=6.9, loc='upper right', frameon=True, facecolor='white')
ax_m3.grid(True, alpha=0.25)
ax_m3.spines['top'].set_visible(False)
ax_m3.spines['right'].set_visible(False)

# Reproducible Physical Boundary Callout Box
add_card(c3_x + 0.18, 2.75, col_w - 0.36, 1.45, fc='#FEF5E7', ec='#D35400', lw=1.2, r=0.10, zorder=2)
ax.text(c3_x + col_w/2, 3.96, "Validated Physical Peri-Plaque Decay Zone", ha='center', va='center', fontsize=9.8, fontweight='bold', color='#A04000', zorder=3)
add_card(c3_x + 0.30, 3.30, 2.08, 0.48, fc='#C0392B', ec='#922B21', lw=1.0, r=0.08, zorder=3)
ax.text(c3_x + 1.34, 3.62, "Cd74 + Ctss: 33.0 µm", ha='center', va='center', fontsize=8.7, fontweight='bold', color='white', zorder=4)
ax.text(c3_x + 1.34, 3.41, "95% CI: 31.0–37.0 µm", ha='center', va='center', fontsize=7.5, color='#FADBD8', zorder=4)

add_card(c3_x + 2.48, 3.30, 2.08, 0.48, fc='#D35400', ec='#A04000', lw=1.0, r=0.08, zorder=3)
ax.text(c3_x + 3.52, 3.62, "10-Gene DAM: 36.0 µm", ha='center', va='center', fontsize=8.7, fontweight='bold', color='white', zorder=4)
ax.text(c3_x + 3.52, 3.41, "95% CI: 34.0–40.0 µm", ha='center', va='center', fontsize=7.5, color='#FDEBD0', zorder=4)

ax.text(c3_x + col_w/2, 3.00, "CP1K normalization eliminates capture-rate bias;\nP2ry12 + Tmem119 shows no monotonic suppression.",
        ha='center', va='center', fontsize=8.0, color='#5D4037', linespacing=1.3, zorder=3)

# 4-Method Benchmark Box
add_card(c3_x + 0.18, 1.24, col_w - 0.36, 1.32, fc='#F4ECF7', ec='#8E44AD', lw=1.2, r=0.10, zorder=2)
ax.text(c3_x + 0.30, 2.34, "4-Method Breakpoint Benchmark (Synthetic MAE)", ha='left', va='center', fontsize=9.4, fontweight='bold', color='#5B2C6F', zorder=3)
bench_txt = (
    "• Segmented Hinge:  0.94 µm MAE  (CosMx: 33.0 µm) [Best]\n"
    "• GAM Cubic Spline: 1.27 µm MAE  (CosMx: 32.8 µm)\n"
    "• PELT (2-Seg):     3.28 µm MAE  (CosMx: 28.0 µm)\n"
    "• Binseg (Step):   12.45 µm MAE  (CosMx: 18.0 µm)"
)
ax.text(c3_x + 0.30, 1.72, bench_txt, ha='left', va='center', fontsize=8.2, family='monospace', color='#2C3E50', linespacing=1.36, zorder=3)

# --- CONNECTING FLOW ARROWS BETWEEN COLUMNS ---
for x_arr in [5.20, 10.49]:
    arr = FancyArrowPatch((x_arr, 4.30), (x_arr + 0.32, 4.30),
                          arrowstyle='Simple,tail_width=5,head_width=14,head_length=10',
                          color='#34495E', zorder=5)
    ax.add_patch(arr)

# --- BOTTOM TAKEAWAY FOOTER ---
add_card(0.25, 0.20, 15.50, 0.66, fc='#2C3E50', ec='#2C3E50', lw=0, r=0.12, zorder=2)
ax.text(8.0, 0.53,
        "CONCLUSION:  Transcriptomic proxy anchors fail in thin cortical sections, whereas physical Aβ immunofluorescence + CP1K normalization + two-tier\n"
        "spillover screening establish a reproducible 33–36 µm peri-plaque microglial activation zone and a spillover-free LRRK1 / L3MBTL4 / PDE4B / UNC5B core.",
        ha='center', va='center', fontsize=9.6, fontweight='bold', color='white', linespacing=1.35, zorder=3)

out_path = os.path.join(FIG_DIR, 'Graphical_Abstract.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.12, facecolor=fig.get_facecolor())
plt.close(fig)
print(f"[SAVED] {out_path}", flush=True)
