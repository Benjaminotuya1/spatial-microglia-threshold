import os
import json
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr, mannwhitneyu, norm
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import statsmodels.api as sm
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

warnings.filterwarnings('ignore')
np.random.seed(42)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.edgecolor': '#222222',
    'axes.linewidth': 0.9,
})

DATA_DIR = os.path.join('Reanalysis_2026', 'data')
FIG_DIR = os.path.join('Reanalysis_2026', 'figures')
RES_DIR = os.path.join('Reanalysis_2026', 'results')
os.makedirs(FIG_DIR, exist_ok=True)

def save_fig(fig, filename):
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.25)
    plt.close(fig)
    print(f"  [SAVED] {filename}", flush=True)

# ==============================================================================
# 0. LOAD ALL PRECOMPUTED SUMMARY FILES
# ==============================================================================
print("=== Loading precomputed results and raw datasets ===", flush=True)
with open(os.path.join(RES_DIR, 'complete_reanalysis_summary.json')) as f:
    sum_base = json.load(f)
with open(os.path.join(RES_DIR, 'critical_audit_and_decontamination.json')) as f:
    audit_res = json.load(f)
with open(os.path.join(RES_DIR, 'advanced_upgrades_summary.json')) as f:
    adv_res = json.load(f)
with open(os.path.join(RES_DIR, 'four_critical_checks_results.json')) as f:
    checks_res = json.load(f)

per_donor_df = pd.read_csv(os.path.join(RES_DIR, 'TableS4_27_Donors_OOF_Performance.csv'))
ablation_df = pd.read_csv(os.path.join(RES_DIR, 'donor_cv_ablation_study.csv'))
sec69_df = pd.read_csv(os.path.join(RES_DIR, 'TableS3_All_69_Cortical_Sections_Proximal_Audit.csv'))
sim_df = pd.read_csv(os.path.join(RES_DIR, 'simulation_study_results.csv'))
bench_df = pd.read_csv(os.path.join(RES_DIR, 'method_benchmark_comparison.csv'))
t_s2 = pd.read_csv(os.path.join(RES_DIR, 'TableS2_Full_140_Gene_Spillover_and_SHAP_Audit.csv'))

# Load SEA-AD 1.89M-cell matrix ONCE
adata_path = os.path.join(DATA_DIR, 'SEAAD_MTG_MERFISH.2024-12-11.h5ad')
adata = sc.read_h5ad(adata_path)
subclasses = adata.obs['Subclass'].astype(str).values
classes = adata.obs['Class'].astype(str).values
tiled_all = adata.obsm['X_spatial_tiled']

# Extract 8-gene expression across ALL 24 subclasses (for SuppFig S8)
s8_genes = ['TACR1', 'PLD5', 'TMEM255A', 'RFX3', 'LRRK1', 'L3MBTL4', 'PDE4B', 'UNC5B']
expr_s8_all = sc.get.obs_df(adata, keys=s8_genes)
expr_s8_all['Subclass'] = subclasses
sub_means_s8 = expr_s8_all.groupby('Subclass')[s8_genes].mean()

# Extract Microglia-PVM subset (195,620 cells)
mg_mask = (subclasses == 'Microglia-PVM')
mg = adata[mg_mask].copy()
valid_genes = sorted([g for g in mg.var_names if not g.startswith('Blank')])
clean_genes = [g for g in valid_genes if g not in ['CD74', 'CTSS']]
expr_mg = sc.get.obs_df(mg, keys=valid_genes)

mg_spatial_pooled = mg.obsm['spatial']
mg_tiled = tiled_all[mg_mask]
act_score = (expr_mg['CD74'] + expr_mg['CTSS']).values
y_gmm = (act_score >= 38.0).astype(int)
q90_thresh = float(np.quantile(act_score, 0.90))
y_q90 = (act_score >= q90_thresh).astype(int)
donors = mg.obs['Donor ID'].astype(str).values
sections = mg.obs['Section'].astype(str).values

# Pooled distance (Top 500 in obsm['spatial'])
top500_idx = np.argsort(act_score)[-500:]
d_pooled, _ = cKDTree(mg_spatial_pooled[top500_idx]).query(mg_spatial_pooled, k=1)

# Within-section distance (Section Top 1% in X_spatial_tiled, non-self d > 0)
df_mg = pd.DataFrame({
    'Donor_ID': donors,
    'Section': sections,
    'Cognitive_Status': mg.obs['Cognitive Status'].astype(str).values,
    'APOE4_Status': mg.obs['APOE4 Status'].astype(str).values,
    'CPS': mg.obs['Continuous Pseudo-progression Score'].astype(float).values,
    'X_Pooled': mg_spatial_pooled[:, 0],
    'Y_Pooled': mg_spatial_pooled[:, 1],
    'X_Tiled': mg_tiled[:, 0],
    'Y_Tiled': mg_tiled[:, 1],
    'Activation_Score': act_score,
    'Event_GMM': y_gmm,
    'Event_Q90': y_q90,
    'Dist_Pooled': d_pooled,
    'LRRK1': expr_mg['LRRK1'].values,
})
sec1pct_list = []
for sec, grp in df_mg.groupby('Section'):
    n_a = max(5, int(round(0.01 * len(grp))))
    sec1pct_list.append(grp.nlargest(n_a, 'Activation_Score')[['X_Tiled', 'Y_Tiled']].values)
anc_s1pct = np.vstack(sec1pct_list)
d_s1, _ = cKDTree(anc_s1pct).query(mg_tiled, k=2)
df_mg['Dist_WithinSec'] = np.where(d_s1[:, 0] == 0, d_s1[:, 1], d_s1[:, 0])

# Load CosMx datasets ONCE (for Fig 6, Fig 8C, SuppFig S7)
dfs_c = []
for gsm in ['GSM8199188', 'GSM8199189']:
    meta = pd.read_csv(os.path.join(DATA_DIR, f'{gsm}_metadata.csv.gz'))
    expr_c = pd.read_csv(os.path.join(DATA_DIR, f'{gsm}_exprMat.csv.gz'))
    gene_cols_all = [c for c in expr_c.columns if c not in ['fov', 'cell_ID'] and not c.startswith('Negative') and not c.startswith('SystemControl')]
    tot_tx = expr_c[gene_cols_all].sum(axis=1).values
    keep_cols = ['fov', 'cell_ID', 'Cd74', 'Ctss', 'Trem2', 'Tyrobp', 'Apoe', 'C1qa', 'C1qb', 'C1qc', 'Hexb', 'Csf1r', 'P2ry12', 'Tmem119', 'Itgax', 'Clec7a', 'Lpl', 'Gpnmb', 'Spp1']
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
        p_cents.append({
            'X_um': np.sum(pgrp['X_um'].values * w) / np.sum(w),
            'Y_um': np.sum(pgrp['Y_um'].values * w) / np.sum(w)
        })
    p_df = pd.DataFrame(p_cents)

    m['MG_Score'] = m['C1qa'] + m['C1qb'] + m['C1qc'] + m['Hexb'] + m['Csf1r'] + m['Tyrobp']
    mg_c = m[m['MG_Score'] >= 4].copy()
    t_cent = cKDTree(p_df[['X_um', 'Y_um']].values)
    d_cent, _ = t_cent.query(mg_c[['X_um', 'Y_um']].values, k=2)
    mg_c['Dist_Plaque_um'] = np.where(d_cent[:, 0] == 0, d_cent[:, 1], d_cent[:, 0])

    tot = np.maximum(mg_c['Total_Transcripts'].values, 1.0)
    mg_c['Raw_CD74_CTSS'] = mg_c['Cd74'] + mg_c['Ctss']
    mg_c['Raw_DAM10'] = mg_c['Cd74'] + mg_c['Ctss'] + mg_c['Trem2'] + mg_c['Tyrobp'] + mg_c['Apoe'] + mg_c['Spp1'] + mg_c['Gpnmb'] + mg_c['Clec7a'] + mg_c['Itgax'] + mg_c['Lpl']
    mg_c['Norm_CD74_CTSS_CP1K'] = 1000.0 * mg_c['Raw_CD74_CTSS'] / tot
    mg_c['Norm_DAM10_CP1K'] = 1000.0 * mg_c['Raw_DAM10'] / tot
    mg_c['Norm_Homeo_CP1K'] = 1000.0 * (mg_c['P2ry12'] + mg_c['Tmem119']) / tot
    dfs_c.append(mg_c)

cosmx_all = pd.concat(dfs_c, ignore_index=True)
c_140 = cosmx_all[(cosmx_all['Dist_Plaque_um'] > 0) & (cosmx_all['Dist_Plaque_um'] <= 140.0)].copy()
bins_c = np.arange(0, 145, 5)
c_140['Bin'] = pd.cut(c_140['Dist_Plaque_um'], bins=bins_c, labels=bins_c[:-1] + 2.5, include_lowest=True).astype(float)
prof_c = c_140.groupby('Bin', observed=False).agg(
    raw_cd74=('Raw_CD74_CTSS', 'mean'),
    se_raw_cd74=('Raw_CD74_CTSS', lambda s: s.std() / np.sqrt(len(s))),
    norm_cd74=('Norm_CD74_CTSS_CP1K', 'mean'),
    se_cd74=('Norm_CD74_CTSS_CP1K', lambda s: s.std() / np.sqrt(len(s))),
    raw_dam10=('Raw_DAM10', 'mean'),
    se_raw_dam10=('Raw_DAM10', lambda s: s.std() / np.sqrt(len(s))),
    norm_dam10=('Norm_DAM10_CP1K', 'mean'),
    se_dam10=('Norm_DAM10_CP1K', lambda s: s.std() / np.sqrt(len(s))),
    norm_homeo=('Norm_Homeo_CP1K', 'mean'),
    se_homeo=('Norm_Homeo_CP1K', lambda s: s.std() / np.sqrt(len(s))),
).dropna()

# ==============================================================================
# FIGURE 1: GMM Activation States
# ==============================================================================
print("Generating Figure 1...", flush=True)
fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=300)
clip_max = 200.0
vals_clip = act_score[act_score <= clip_max]
ax.hist(vals_clip, bins=75, density=True, color='#BDC3C7', edgecolor='#7F8C8D', alpha=0.65, label='Observed Microglia-PVM (n = 195,620)')

gmm = GaussianMixture(n_components=2, random_state=42).fit(act_score.reshape(-1, 1))
order_g = np.argsort(gmm.means_.ravel())
x_grid = np.linspace(0, clip_max, 600)
w0, m0, s0 = gmm.weights_[order_g[0]], gmm.means_[order_g[0], 0], np.sqrt(gmm.covariances_[order_g[0], 0, 0])
w1, m1, s1 = gmm.weights_[order_g[1]], gmm.means_[order_g[1], 0], np.sqrt(gmm.covariances_[order_g[1], 0, 0])
pdf0 = w0 * norm.pdf(x_grid, m0, s0)
pdf1 = w1 * norm.pdf(x_grid, m1, s1)
ax.plot(x_grid, pdf0, color='#2980B9', lw=2.4, label=f'GMM Class 0: Homeostatic / Baseline (μ = {m0:.1f})')
ax.plot(x_grid, pdf1, color='#C0392B', lw=2.4, label=f'GMM Class 1: Activated / DAM-like (μ = {m1:.1f})')
ax.plot(x_grid, pdf0 + pdf1, color='#2C3E50', ls='--', lw=1.6, alpha=0.8, label='Combined 2-Component GMM Density')

pct_c1 = 100.0 * float(np.mean(act_score >= 38.0))
pct_q90 = 100.0 * float(np.mean(act_score >= 107.0))
ax.axvline(38.0, color='#C0392B', ls='-', lw=2.2)
ax.axvline(107.0, color='#8E44AD', ls=':', lw=2.2)

ax.annotate(f"GMM Decision Boundary\n38.0 spots ({pct_c1:.2f}% Class 1)",
            xy=(38.0, ax.get_ylim()[1] * 0.62), xytext=(55.0, ax.get_ylim()[1] * 0.72),
            fontsize=9.2, fontweight='bold', color='#922B21',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#FDEDEC', edgecolor='#C0392B'),
            arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.5))

ax.annotate(f"90th Percentile Cutoff\n107.0 spots ({pct_q90:.2f}% extreme)",
            xy=(107.0, ax.get_ylim()[1] * 0.25), xytext=(122.0, ax.get_ylim()[1] * 0.42),
            fontsize=9.2, fontweight='bold', color='#5B2C6F',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#F5EEF8', edgecolor='#8E44AD'),
            arrowprops=dict(arrowstyle='->', color='#8E44AD', lw=1.5))

ax.set_xlim(0, clip_max)
ax.set_xlabel("Composite Activation Score (CD74 + CTSS Volume-Scaled Linear Spot Counts)", fontsize=10.8)
ax.set_ylabel("Probability Density", fontsize=10.8)
ax.set_title("Figure 1. Two-Component Gaussian Mixture Decomposition of Microglial Activation States\n(SEA-AD Human MTG MERFISH, n = 195,620 Microglia-PVM Across 27 Donors)",
             fontsize=11.8, fontweight='bold', pad=12)
ax.legend(fontsize=8.8, loc='upper right', frameon=True, facecolor='white', framealpha=0.95)
ax.grid(True, alpha=0.25)
save_fig(fig, 'Figure_1_GMM_Activation_States.png')

# ==============================================================================
# FIGURE 2: Donor-Level Generalization (ROC + Ablation Inset) & 27-Donor Bars
# ==============================================================================
print("Generating Figure 2...", flush=True)
X_138_dist = expr_mg[clean_genes].copy()
X_138_dist['Distance_to_Plaque_um'] = d_pooled

# Compute 5-fold GroupKFold OOF probabilities & Random 80/20 probabilities for exact ROC curves
gkf = GroupKFold(n_splits=5)
oof_probs = np.zeros(len(y_gmm))
for fold, (tr, te) in enumerate(gkf.split(X_138_dist, y_gmm, groups=donors)):
    clf_f = xgb.XGBClassifier(objective='binary:logistic', tree_method='hist',
                              learning_rate=0.05, max_depth=6, n_estimators=100,
                              random_state=42 + fold, n_jobs=-1)
    clf_f.fit(X_138_dist.iloc[tr], y_gmm[tr])
    oof_probs[te] = clf_f.predict_proba(X_138_dist.iloc[te])[:, 1]

X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_138_dist, y_gmm, test_size=0.20, random_state=42, stratify=y_gmm)
clf_rand = xgb.XGBClassifier(objective='binary:logistic', tree_method='hist',
                             learning_rate=0.05, max_depth=6, n_estimators=100,
                             random_state=42, n_jobs=-1)
clf_rand.fit(X_tr_r, y_tr_r)
rand_probs = clf_rand.predict_proba(X_te_r)[:, 1]

fpr_gkf, tpr_gkf, _ = roc_curve(y_gmm, oof_probs)
fpr_rnd, tpr_rnd, _ = roc_curve(y_te_r, rand_probs)

fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), dpi=300, gridspec_kw={'width_ratios': [1.0, 1.35]})

# Panel A: ROC Curves + Legible Ablation Inset
ax = axes[0]
ax.plot(fpr_rnd, tpr_rnd, color='#C0392B', ls='--', lw=2.2, label='Random 80/20 Cell Split (AUC = 0.8964)')
ax.plot(fpr_gkf, tpr_gkf, color='#1B4F72', ls='-', lw=2.5, label='5-Fold Donor GroupKFold (Mean = 0.8910 ± 0.0113)')
ax.plot([0, 1], [0, 1], color='#999999', ls=':', lw=1.2)
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.02)
ax.set_xlabel("False Positive Rate", fontsize=10.8)
ax.set_ylabel("True Positive Rate", fontsize=10.8)
ax.set_title("A. Donor-Level Cross-Validation ROC & Feature Ablation", fontsize=11.8, fontweight='bold', pad=10)
ax.legend(fontsize=8.8, loc='upper left', frameon=True, facecolor='white', framealpha=0.95)
ax.grid(True, alpha=0.25)

# Legible Inset Bar Chart (Table 1 Ablation)
ax_in = inset_axes(ax, width="52%", height="40%", loc='lower right', bbox_to_anchor=(0, 0.06, 0.98, 0.94), bbox_transform=ax.transAxes)
ab_labels = ['Genes Only\n(138)', 'Genes +\nWithinSec Dist', 'Distance Only\n(WithinSec)']
ab_vals = [0.8912, 0.8917, 0.5568]
ab_cols = ['#2E86C1', '#117A65', '#95A5A6']
b_in = ax_in.bar(range(3), ab_vals, color=ab_cols, edgecolor='#222222', width=0.58)
for b, v in zip(b_in, ab_vals):
    ax_in.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.4f}", ha='center', va='bottom', fontsize=7.6, fontweight='bold')
ax_in.set_xticks(range(3))
ax_in.set_xticklabels(ab_labels, fontsize=7.2)
ax_in.set_ylim(0.45, 1.03)
ax_in.set_ylabel("5-Fold AUC", fontsize=7.8)
ax_in.set_title("Feature Ablation (Table 1)", fontsize=8.3, fontweight='bold', pad=4)
ax_in.tick_params(axis='y', labelsize=7.5)
ax_in.grid(True, alpha=0.25, axis='y')

# Panel B: Vertical Bar Chart of All 27 Donors (Mega-Donors Highlighted!)
ax2 = axes[1]
d_sorted = per_donor_df.sort_values('ROC_AUC_138', ascending=True).reset_index(drop=True)
mega_donors = {'H20.33.004', 'H20.33.001', 'H21.33.031', 'H20.33.015'}
bar_colors = ['#C0392B' if d in mega_donors else '#457B9D' for d in d_sorted['Donor_ID']]
x_d = np.arange(len(d_sorted))
ax2.bar(x_d, d_sorted['ROC_AUC_138'], color=bar_colors, edgecolor='#1D3557', width=0.72, alpha=0.90)

med_27 = float(np.median(d_sorted['ROC_AUC_138']))
ax2.axhline(med_27, color='#E67E22', ls='--', lw=2.2, label=f'27-Donor Median ({med_27:.4f}; IQR 0.8498–0.8970)')
ax2.axhline(0.8910, color='#1B365D', ls=':', lw=2.4, label='5-Fold Cell-Weighted Mean (0.8910)')

ax2.set_xticks(x_d)
ax2.set_xticklabels(d_sorted['Donor_ID'], rotation=65, ha='right', fontsize=7.8)
ax2.set_ylim(0.74, 0.95)
ax2.set_ylabel("Held-Out Donor ROC-AUC", fontsize=10.8)
ax2.set_xlabel("Donor ID (Sorted by Out-of-Fold ROC-AUC)", fontsize=10.5)
ax2.set_title("B. Out-of-Fold ROC-AUC Across All 27 Donors (4 Mega-Donors Highlighted)", fontsize=11.8, fontweight='bold', pad=10)
ax2.grid(True, alpha=0.25, axis='y')

leg_2b = [
    Patch(facecolor='#C0392B', edgecolor='#1D3557', label='4 Mega-Donors (H20.33.004, H20.33.001, H21.33.031, H20.33.015)'),
    Patch(facecolor='#457B9D', edgecolor='#1D3557', label='Remaining 23 Donors'),
    Line2D([0], [0], color='#1B365D', ls=':', lw=2.4, label='5-Fold Cell-Weighted Mean (0.8910)'),
    Line2D([0], [0], color='#E67E22', ls='--', lw=2.2, label=f'27-Donor Median ({med_27:.4f})'),
]
ax2.legend(handles=leg_2b, fontsize=8.1, loc='upper left', frameon=True, facecolor='white', framealpha=0.95)

fig.tight_layout(w_pad=2.5)
save_fig(fig, 'Figure_2_Donor_Generalization_and_Ablation.png')

# ==============================================================================
# FIGURE 3: SHAP Landscape (Unscreened 138-Gene) + SUPP FIG S1 (Pre-Exclusion)
# ==============================================================================
print("Generating Figure 3 & Supplementary Figure S1...", flush=True)
rng_s = np.random.default_rng(42)
sub_idx_3 = rng_s.choice(len(y_gmm), size=3500, replace=False)

# Train full 138-gene clean model (also used for Fig 7D!)
X_clean_ws = expr_mg[clean_genes].copy()
X_clean_ws['Distance_WithinSection_um'] = df_mg['Dist_WithinSec'].values
clf_clean = xgb.XGBClassifier(objective='binary:logistic', tree_method='hist',
                              learning_rate=0.05, max_depth=6, n_estimators=100,
                              random_state=42, n_jobs=-1)
clf_clean.fit(X_clean_ws, y_gmm)
exp_clean = shap.TreeExplainer(clf_clean)
sv_clean_3 = exp_clean.shap_values(X_clean_ws.iloc[sub_idx_3])

plt.figure(figsize=(10.5, 7.2), dpi=300)
shap.summary_plot(sv_clean_3, X_clean_ws.iloc[sub_idx_3], max_display=15, show=False, plot_size=(10.5, 7.2))
fig_3 = plt.gcf()
ax_3 = plt.gca()
ax_3.set_title("Figure 3. Unscreened 138-Gene SHAP Feature Importance Landscape\n(CD74 and CTSS Excluded; Top 15 Transcriptomic Predictors of Microglial Activation)",
               fontsize=11.5, fontweight='bold', pad=14)
ax_3.set_xlabel("SHAP Value (Impact on Activation Log-Odds)", fontsize=10.5)
fig_3.subplots_adjust(top=0.88, bottom=0.10, left=0.18, right=0.88)
save_fig(fig_3, 'Figure_3_SHAP_Landscape_138Gene.png')

# SuppFig S1: Pre-Exclusion Tautological SHAP (CD74 & CTSS Retained)
X_circ = expr_mg[valid_genes].copy()
clf_circ = xgb.XGBClassifier(objective='binary:logistic', tree_method='hist',
                             learning_rate=0.05, max_depth=6, n_estimators=100,
                             random_state=42, n_jobs=-1)
clf_circ.fit(X_circ, y_gmm)
sv_circ = shap.TreeExplainer(clf_circ).shap_values(X_circ.iloc[sub_idx_3])

plt.figure(figsize=(10.5, 7.2), dpi=300)
shap.summary_plot(sv_circ, X_circ.iloc[sub_idx_3], max_display=15, show=False, plot_size=(10.5, 7.2))
fig_s1 = plt.gcf()
ax_s1 = plt.gca()
ax_s1.set_title("Supplementary Figure S1. Pre-Exclusion Tautological SHAP Control\n(Target Markers CD74 and CTSS Retained, Confirming Target-Leakage Dominance)",
                fontsize=11.5, fontweight='bold', pad=14)
ax_s1.set_xlabel("SHAP Value (Impact on Activation Log-Odds)", fontsize=10.5)
fig_s1.subplots_adjust(top=0.88, bottom=0.10, left=0.18, right=0.88)
save_fig(fig_s1, 'SuppFig_S1_PreExclusion_Tautological_SHAP.png')

# ==============================================================================
# FIGURE 4: Two-Tier Spillover Audit (Top 10) + 57-Section Proxy Instability Strip
# ==============================================================================
print("Generating Figure 4...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), dpi=300)

# Panel A: Top 10 Mean |SHAP| Genes with 3 Explicit Spillover Categories
ax = axes[0]
top12_shap = audit_res['spillover_decontamination_cv']['Baseline_138_Genes']['top_12_shap']
genes_10 = list(top12_shap.keys())[:10][::-1]
vals_10 = [top12_shap[g] for g in genes_10]

tier_map = {
    'LRRK1':    ('Intrinsic', '0.87× max'),
    'L3MBTL4':  ('Intrinsic', '1.41× max'),
    'PDE4B':    ('Intrinsic', '1.42× max'),
    'UNC5B':    ('Intrinsic', '1.32× max'),
    'ID3':      ('Intrinsic', '<1.5× all'),
    'RFX3':     ('Tier2',     '1.60× Endothelial'),
    'TMEM255A': ('Tier2',     '1.75× OPC'),
    'TACR1':    ('Tier2',     '2.32× Sst Chodl'),
    'PLD5':     ('Tier2',     '2.17× Sst Chodl'),
    'RBFOX3':   ('Tier1',     '2.48× Neuron'),
}
col_map = {'Intrinsic': '#C0392B', 'Tier2': '#E67E22', 'Tier1': '#7F8C8D'}
hat_map = {'Intrinsic': '', 'Tier2': '//', 'Tier1': 'xx'}

y_p = np.arange(len(genes_10))
for i, (g, v) in enumerate(zip(genes_10, vals_10)):
    t_cat, note = tier_map[g]
    ax.barh(i, v, color=col_map[t_cat], edgecolor='#222222', hatch=hat_map[t_cat], alpha=0.90, height=0.68)
    lbl = f"{v:.3f}  ({g}: {note})" if t_cat == 'Tier2' else f"{v:.3f}  [{note}]"
    ax.text(v + 0.015, i, lbl, va='center', ha='left', fontsize=8.4,
            fontweight='bold' if t_cat in ('Intrinsic', 'Tier2') else 'normal')

ax.set_yticks(y_p)
ax.set_yticklabels(genes_10, fontsize=9.8, fontweight='bold')
ax.set_xlim(0, 1.05)
ax.set_xlabel("Mean |SHAP Value| (Unscreened 138-Gene Donor-CV Model)", fontsize=10.5)
ax.set_title("A. Two-Tier Neighbor Spillover Audit of Top 10 SHAP Drivers\n(1.89M Cells Across All 24 SEA-AD Subclasses)", fontsize=11.5, fontweight='bold', pad=10)
ax.grid(True, alpha=0.25, axis='x')

leg_4a = [
    Patch(facecolor='#C0392B', edgecolor='#222222', label='Spillover-Resistant Intrinsic Myeloid (<1.5× all 24 subclasses)'),
    Patch(facecolor='#E67E22', edgecolor='#222222', hatch='//', label='Tier-2 Rare-Subclass Spillover (TACR1, PLD5, TMEM255A, RFX3)'),
    Patch(facecolor='#7F8C8D', edgecolor='#222222', hatch='xx', label='Tier-1 Broad-Class Spillover (RBFOX3; plus GRIN2A, GRIP2, CD22, MOG)'),
]
ax.legend(handles=leg_4a, fontsize=8.0, loc='lower right', frameon=True, facecolor='white', framealpha=0.95)

# Panel B: Scatter/Strip Plot of Within-Section Hinge Knots Across 57 Evaluable Sections
ax2 = axes[1]
eval_sec = sec69_df[sec69_df['Hinge_Knot_um'] != 'Underpowered (<50)'].copy()
eval_sec['Hinge_Knot_num'] = eval_sec['Hinge_Knot_um'].astype(float)
eval_sec = eval_sec.sort_values('Hinge_Knot_num').reset_index(drop=True)
x_sec = np.arange(1, len(eval_sec) + 1)
y_knots = eval_sec['Hinge_Knot_num'].values
med_h = float(np.median(y_knots))
q25_h, q75_h = float(np.percentile(y_knots, 25)), float(np.percentile(y_knots, 75))

ax2.axhspan(q25_h, q75_h, color='#AED6F1', alpha=0.35, label=f'SEA-AD Within-Section IQR ({q25_h:.0f}–{q75_h:.0f} µm)')
ax2.axhline(med_h, color='#1B4F72', ls='--', lw=2.0, label=f'SEA-AD Within-Section Median ({med_h:.0f} µm)')

# Overlay CosMx CP1K Physical Plaque Band (33 um, CI 31-37) & 10-Gene DAM Band (36 um, CI 34-40)
ax2.axhspan(31.0, 37.0, color='#C0392B', alpha=0.22, label='CosMx CP1K Physical Plaque Boundary: 33 µm (95% CI 31–37)')
ax2.axhline(33.0, color='#C0392B', ls='-', lw=2.2)
ax2.axhspan(34.0, 40.0, color='#E67E22', alpha=0.20, label='CosMx 10-Gene DAM Boundary: 36 µm (95% CI 34–40)')
ax2.axhline(36.0, color='#D35400', ls=':', lw=2.0)

ax2.scatter(x_sec, y_knots, color='#2E86C1', edgecolor='#1B4F72', s=42, alpha=0.90, zorder=4, label='Individual SEA-AD Sections (n = 57 evaluable)')
ax2.set_xlim(0, len(eval_sec) + 1)
ax2.set_ylim(5, 84)
ax2.set_xlabel("Evaluable SEA-AD Cortical Sections (n = 57, Sorted by Knot)", fontsize=10.5)
ax2.set_ylabel("Estimated Spatial Decay Breakpoint (µm)", fontsize=10.5)
ax2.set_title("B. Within-Section Proxy Knot Scatter Across 57 Cortical Sections\nvs. Direct Physical Aβ Plaque Boundaries in CosMx", fontsize=11.5, fontweight='bold', pad=10)
ax2.legend(fontsize=8.0, loc='upper left', frameon=True, facecolor='white', framealpha=0.95)
ax2.grid(True, alpha=0.25)

fig.tight_layout(w_pad=2.5)
save_fig(fig, 'Figure_4_TwoTier_Spillover_and_Section_Instability.png')

# ==============================================================================
# FIGURE 5: Negative-Control Anchor Specificity (Pooled Coordinates)
# ==============================================================================
print("Generating Figure 5...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), dpi=300)

# Panel A: Bar chart of <25 um elevation across anchor classes
ax = axes[0]
ctrl_labels = ['Top-500\nActivation\nAnchors', '100 Random\nPermutations\n(Mean ± SD)', 'TACR1\nTop-500', 'GRIP2\nTop-500', 'CD22\nTop-500', 'Composite\nHomeostatic\nTop-500']
ctrl_vals = [1.33, -0.089, -0.59, -1.85, 0.21, -0.53]
ctrl_errs = [0.0, 0.573, 0.0, 0.0, 0.0, 0.0]
ctrl_cols = ['#C0392B', '#7F8C8D', '#2E86C1', '#2E86C1', '#5DADE2', '#1B4F72']
x_c = np.arange(len(ctrl_labels))
bars_5 = ax.bar(x_c, ctrl_vals, yerr=ctrl_errs, capsize=5, color=ctrl_cols, edgecolor='#222222', width=0.58, alpha=0.88)
ax.axhline(0, color='#222222', lw=1.0)
for b, v in zip(bars_5, ctrl_vals):
    va = 'bottom' if v >= 0 else 'top'
    off = 0.08 if v >= 0 else -0.08
    ax.text(b.get_x() + b.get_width()/2, v + off, f"{v:+.2f}%", ha='center', va=va, fontsize=8.8, fontweight='bold')
ax.set_xticks(x_c)
ax.set_xticklabels(ctrl_labels, fontsize=8.6)
ax.set_ylim(-2.4, 2.0)
ax.set_ylabel("Proximal (<25 µm) Elevation in Extreme Activation (%)", fontsize=10.2)
ax.set_title("A. Anchor Specificity Comparison (Pooled Coordinates)", fontsize=11.5, fontweight='bold', pad=10)
ax.grid(True, alpha=0.25, axis='y')

# Panel B: Permutation null distribution (mean -0.089%, SD 0.573%, observed +1.33%, p = 0.0198)
ax2 = axes[1]
rng_p = np.random.default_rng(123)
x_null = np.linspace(-2.0, 2.0, 400)
y_null = norm.pdf(x_null, -0.089, 0.573)
ax2.fill_between(x_null, 0, y_null, color='#BDC3C7', alpha=0.55, label='100 Random Anchor Permutations\n(Mean = −0.089% ± 0.573% SD)')
ax2.plot(x_null, y_null, color='#555555', lw=2.0)
ax2.axvline(1.33, color='#C0392B', ls='--', lw=2.5, label='Observed Top-500 Activation Anchors (+1.33%)\nEmpirical p = 0.0198')
ax2.axvline(-0.53, color='#1B4F72', ls=':', lw=2.0, label='Composite Homeostatic Control (−0.53%)')
ax2.set_xlabel("Proximal (<25 µm) Elevation in Extreme Activation (%)", fontsize=10.2)
ax2.set_ylabel("Null Density", fontsize=10.5)
ax2.set_title("B. Empirical Permutation Test of Pooled Anchor Specificity", fontsize=11.5, fontweight='bold', pad=10)
ax2.legend(fontsize=8.6, loc='upper left', frameon=True, facecolor='white', framealpha=0.95)
ax2.grid(True, alpha=0.25)

fig.text(0.5, -0.02, "Note: Computed in pooled obsm['spatial'] coordinates; supports pooled anchor specificity only (not within-section spatial boundary recovery).",
         ha='center', fontsize=8.8, style='italic', color='#444444')
fig.tight_layout(w_pad=2.5)
save_fig(fig, 'Figure_5_Negative_Control_Anchor_Specificity.png')

# ==============================================================================
# FIGURE 6: CosMx Physical Plaque Validation (CP1K Normalized, 0–60 um + Inset)
# ==============================================================================
print("Generating Figure 6...", flush=True)
prof_60 = prof_c[prof_c.index <= 60.0].copy()
bx_all = prof_c.index.values
bx_60 = prof_60.index.values

k_cd74 = checks_res['check_4_cosmx_normalized']['norm_cd74_knot_um']
ci_cd74 = checks_res['check_4_cosmx_normalized']['norm_cd74_ci_95']
k_dam10 = checks_res['check_4_cosmx_normalized']['norm_dam10_knot_um']
ci_dam10 = checks_res['check_4_cosmx_normalized']['norm_dam10_ci_95']

# Fit segmented hinge on full 0-140 um profile, evaluate over 0-60 um
X_fit1_all = np.column_stack([np.ones_like(bx_all), np.minimum(bx_all, k_cd74), np.maximum(0, bx_all - k_cd74)])
m1_fit = sm.OLS(prof_c['norm_cd74'].values, X_fit1_all).fit()
X_eval1 = np.column_stack([np.ones_like(bx_60), np.minimum(bx_60, k_cd74), np.maximum(0, bx_60 - k_cd74)])
y_pred1_60 = m1_fit.predict(X_eval1)

X_fit2_all = np.column_stack([np.ones_like(bx_all), np.minimum(bx_all, k_dam10), np.maximum(0, bx_all - k_dam10)])
m2_fit = sm.OLS(prof_c['norm_dam10'].values, X_fit2_all).fit()
X_eval2 = np.column_stack([np.ones_like(bx_60), np.minimum(bx_60, k_dam10), np.maximum(0, bx_60 - k_dam10)])
y_pred2_60 = m2_fit.predict(X_eval2)

fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.0), dpi=300)

# Panel A: CP1K Cd74+Ctss (0-60 um) + Raw secondary line
ax = axes[0]
l1 = ax.errorbar(bx_60, prof_60['norm_cd74'].values, yerr=prof_60['se_cd74'].values, fmt='o-', color='#C0392B',
                 ecolor='#E6B0AA', elinewidth=1.4, capsize=2.5, lw=2.5, markersize=6, label='CP1K-Normalized Cd74 + Ctss (± SE)')
l2, = ax.plot(bx_60, y_pred1_60, 'k--', lw=2.2, label=f'Two-Piece Hinge Fit (Breakpoint = {k_cd74:.1f} µm)')
ax.axvline(k_cd74, color='#117A65', ls=':', lw=2.4)
p_ci1 = ax.axvspan(ci_cd74[0], ci_cd74[1], color='#117A65', alpha=0.16, label=f'95% Plaque-Cluster CI ({ci_cd74[0]:.1f}–{ci_cd74[1]:.1f} µm)')

ax_r1 = ax.twinx()
l3, = ax_r1.plot(bx_60, prof_60['raw_cd74'].values, 's--', color='#7F8C8D', alpha=0.55, lw=1.6, markersize=4, label='Raw Counts (secondary axis)')
ax_r1.set_ylabel("Raw Cd74 + Ctss Counts (Unnormalized)", color='#7F8C8D', fontsize=9.8)
ax_r1.tick_params(axis='y', labelcolor='#7F8C8D')

ax.set_xlim(0, 62)
ax.set_xlabel("Distance to Nearest Anti-Aβ Plaque Centroid (µm, 5 µm bins)", fontsize=10.5)
ax.set_ylabel("Counts per 1,000 total transcripts (CP1K) ± SE", fontsize=10.5)
ax.set_title("A. Direct Aβ Plaque Validation (402 Plaques, d=0 Excl.)\nCP1K-Normalized Cd74 + Ctss (Breakpoint = 33.0 µm)", fontsize=11.5, fontweight='bold', pad=10)
ax.grid(True, alpha=0.25)

h1, lb1 = ax.get_legend_handles_labels()
h2, lb2 = ax_r1.get_legend_handles_labels()
ax.legend(h1 + h2, lb1 + lb2, fontsize=8.3, loc='upper right', frameon=True, facecolor='white', framealpha=0.95)

# Panel B: CP1K 10-Gene DAM (0-60 um) + Raw secondary line + P2ry12+Tmem119 Inset
ax2 = axes[1]
ax2.errorbar(bx_60, prof_60['norm_dam10'].values, yerr=prof_60['se_dam10'].values, fmt='s-', color='#D35400',
             ecolor='#F5CBA7', elinewidth=1.4, capsize=2.5, lw=2.5, markersize=6, label='CP1K-Normalized 10-Gene DAM (± SE)')
ax2.plot(bx_60, y_pred2_60, 'k--', lw=2.2, label=f'Two-Piece Hinge Fit (Breakpoint = {k_dam10:.1f} µm)')
ax2.axvline(k_dam10, color='#117A65', ls=':', lw=2.4)
ax2.axvspan(ci_dam10[0], ci_dam10[1], color='#D35400', alpha=0.14, label=f'95% Plaque-Cluster CI ({ci_dam10[0]:.1f}–{ci_dam10[1]:.1f} µm)')

ax_r2 = ax2.twinx()
ax_r2.plot(bx_60, prof_60['raw_dam10'].values, '^--', color='#7F8C8D', alpha=0.50, lw=1.6, markersize=4, label='Raw Counts (secondary axis)')
ax_r2.set_ylabel("Raw 10-Gene DAM Counts (Unnormalized)", color='#7F8C8D', fontsize=9.8)
ax_r2.tick_params(axis='y', labelcolor='#7F8C8D')

ax2.set_xlim(0, 62)
ax2.set_xlabel("Distance to Nearest Anti-Aβ Plaque Centroid (µm, 5 µm bins)", fontsize=10.5)
ax2.set_ylabel("Counts per 1,000 total transcripts (CP1K) ± SE", fontsize=10.5)
ax2.set_title("B. Direct Aβ Plaque Validation (GSE263791 CosMx)\nCP1K-Normalized 10-Gene DAM (Breakpoint = 36.0 µm)", fontsize=11.5, fontweight='bold', pad=10)
ax2.grid(True, alpha=0.25)

h1b, lb1b = ax2.get_legend_handles_labels()
h2b, lb2b = ax_r2.get_legend_handles_labels()
ax2.legend(h1b + h2b, lb1b + lb2b, fontsize=8.1, loc='upper right', frameon=True, facecolor='white', framealpha=0.95)

# Inset for P2ry12 + Tmem119 CP1K (cleanly in middle-right below legend)
ax_in6 = inset_axes(ax2, width="45%", height="34%", loc='center right', borderpad=1.6)
ax_in6.errorbar(bx_60, prof_60['norm_homeo'].values, yerr=prof_60['se_homeo'].values, fmt='o-', color='#2980B9',
                ecolor='#AED6F1', lw=1.6, markersize=3.5, capsize=1.5)
ax_in6.set_title("P2ry12 + Tmem119 (CP1K)", fontsize=8.0, fontweight='bold', pad=3)
ax_in6.set_xlabel("Distance (µm)", fontsize=7.2)
ax_in6.set_ylabel("CP1K", fontsize=7.2)
ax_in6.tick_params(labelsize=6.8)
ax_in6.set_ylim(0.5, 3.8)
ax_in6.grid(True, alpha=0.25)
ax_in6.text(0.5, 0.08, "No monotonic suppression:\n2.86 (0–5 µm) vs. 1.97 (50–140 µm)",
            transform=ax_in6.transAxes, fontsize=6.4, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#EBF5FB', edgecolor='#2980B9', alpha=0.92))

fig.tight_layout(w_pad=3.2)
save_fig(fig, 'Figure_6_CosMx_Physical_Plaque_Validation_CP1K.png')

# ==============================================================================
# FIGURE 7: Deconvolution, Pseudoreplication, Stacked Niche, 2D SHAP LRRK1
# ==============================================================================
print("Generating Figure 7...", flush=True)
# Donor-level within-section Delta_Score vs CPS
donor_rows = []
for d_id, grp in df_mg.groupby('Donor_ID'):
    p25 = grp[(grp['Dist_WithinSec'] > 0) & (grp['Dist_WithinSec'] < 25)]
    d25 = grp[(grp['Dist_WithinSec'] >= 25) & (grp['Dist_WithinSec'] <= 140)]
    donor_rows.append({
        'Donor_ID': d_id,
        'Cog': grp['Cognitive_Status'].iloc[0],
        'APOE4': grp['APOE4_Status'].iloc[0],
        'CPS': float(grp['CPS'].iloc[0]),
        'n_lt25': len(p25),
        'Delta_Score': float(p25['Activation_Score'].mean() - d25['Activation_Score'].mean()) if (len(p25) > 0 and len(d25) > 0) else np.nan
    })
ddf = pd.DataFrame(donor_rows).dropna()
rho_cps, p_cps = spearmanr(ddf['CPS'], ddf['Delta_Score'])
_, p_cog = mannwhitneyu(ddf[ddf['Cog'] == 'Dementia']['Delta_Score'], ddf[ddf['Cog'] == 'No dementia']['Delta_Score'], alternative='two-sided')
_, p_apoe = mannwhitneyu(ddf[ddf['APOE4'] == 'Y']['Delta_Score'], ddf[ddf['APOE4'] == 'N']['Delta_Score'], alternative='two-sided')

# Strictly within-section multicellular niche (Global Top-500 in X_spatial_tiled)
anc_g500 = mg_tiled[top500_idx]
d_all_tiled, _ = cKDTree(anc_g500).query(tiled_all, k=1)
is_neuron_all = np.isin(classes, ['Neuronal: Glutamatergic', 'Neuronal: GABAergic'])

niche_rows = []
for z_label, z_mask in [
    ('Inner Halo\n(0–25 µm)', (d_all_tiled > 0) & (d_all_tiled <= 25)),
    ('Warning Zone\n(25–50 µm)', (d_all_tiled > 25) & (d_all_tiled <= 50)),
    ('Distal\n(50–150 µm)', (d_all_tiled > 50) & (d_all_tiled <= 150)),
]:
    sc_vc = pd.Series(subclasses[z_mask]).value_counts(normalize=True) * 100.0
    mg_p = sc_vc.get('Microglia-PVM', 0.0)
    neu_p = 100.0 * float(np.mean(is_neuron_all[z_mask]))
    oli_p = sc_vc.get('Oligodendrocyte', 0.0)
    ast_p = sc_vc.get('Astrocyte', 0.0)
    oth_p = max(0.0, 100.0 - (mg_p + neu_p + oli_p + ast_p))
    niche_rows.append({'Zone': z_label, 'Microglia-PVM': mg_p, 'Neurons': neu_p, 'Oligodendrocyte': oli_p, 'Astrocyte': ast_p, 'Other': oth_p})
niche_df = pd.DataFrame(niche_rows)

# 2D SHAP LRRK1 subsample within <=140 um
sub_idx_7d = rng_s.choice(np.where(df_mg['Dist_WithinSec'].values <= 140)[0], size=4000, replace=False)
sv_7d = exp_clean.shap_values(X_clean_ws.iloc[sub_idx_7d])
lrrk1_col = list(X_clean_ws.columns).index('LRRK1')
shap_lrrk1 = sv_7d[:, lrrk1_col]
dist_7d = df_mg['Dist_WithinSec'].values[sub_idx_7d]
expr_lrrk1_7d = df_mg['LRRK1'].values[sub_idx_7d]

fig, axes = plt.subplots(2, 2, figsize=(15.5, 12.2), dpi=300)

# Panel A: Supertype Proportions + Mean Activation Score Annotated
ax = axes[0, 0]
sup_df = pd.DataFrame(adv_res['upgrade_1_supertypes']).sort_values('pct_of_total', ascending=True)
sup_cols = ['#C0392B' if 'Micro-PVM_2_3' in s or 'Micro-PVM_3' in s else ('#E67E22' if 'Monocyte' in s else '#5DADE2') for s in sup_df['Supertype']]
bars_7a = ax.barh(sup_df['Supertype'], sup_df['pct_of_total'], color=sup_cols, edgecolor='#222222', height=0.65, alpha=0.90)
for b, row in zip(bars_7a, sup_df.itertuples()):
    ax.text(b.get_width() + 0.9, b.get_y() + b.get_height()/2,
            f"{row.pct_of_total:.1f}%  |  Mean Score = {row.mean_activation_score:.1f}",
            va='center', ha='left', fontsize=8.8, fontweight='bold' if row.pct_of_total > 8 else 'normal')
ax.set_xlim(0, 96)
ax.set_xlabel("Proportion of Myeloid Cells (%)", fontsize=10.5)
ax.set_title("A. Myeloid Supertype Deconvolution\n(Proportions & Mean CD74+CTSS Activation Score)", fontsize=11.5, fontweight='bold', pad=10)
ax.grid(True, alpha=0.25, axis='x')

# Panel B: 27-Donor Pseudoreplication & CPS Scatter (H20.33.004 Star!)
ax = axes[0, 1]
for _, r in ddf.iterrows():
    c = '#C0392B' if r['Cog'] == 'Dementia' else '#2980B9'
    if r['Donor_ID'] == 'H20.33.004':
        ax.scatter(r['CPS'], r['Delta_Score'], s=340, c='#C0392B', marker='*', edgecolor='#111111', linewidth=1.1, zorder=5)
    else:
        mk = '^' if r['APOE4'] == 'Y' else 'o'
        ax.scatter(r['CPS'], r['Delta_Score'], s=72, c=c, marker=mk, edgecolor='#111111', linewidth=0.8, alpha=0.88, zorder=3)

z_fit = np.polyfit(ddf['CPS'], ddf['Delta_Score'], 1)
x_line = np.linspace(ddf['CPS'].min() - 0.03, ddf['CPS'].max() + 0.03, 100)
ax.plot(x_line, np.polyval(z_fit, x_line), color='#2C3E50', ls='--', lw=2.0, zorder=2)
ax.axhline(0, color='#7F8C8D', ls='--', lw=1.2)

mega = ddf[ddf['Donor_ID'] == 'H20.33.004'].iloc[0]
ax.set_ylim(-44, 26)
ax.set_xlim(0.08, 0.96)
ax.annotate(
    "Mega-Donor H20.33.004\n65.5% of dementia proximal cells,\n82.0% of APOE4+ proximal cells",
    xy=(mega['CPS'] - 0.025, mega['Delta_Score']),
    xytext=(0.11, 2.0),
    fontsize=8.2, fontweight='bold', color='#922B21',
    bbox=dict(boxstyle='round,pad=0.35', facecolor='#FDEDEC', edgecolor='#C0392B', alpha=0.95),
    arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.6, connectionstyle='arc3,rad=-0.12')
)

ax.set_xlabel("Donor Continuous Pseudo-progression Score (CPS)", fontsize=10.5)
ax.set_ylabel("Within-Section Donor ΔScore (<25 µm minus 25–140 µm)", fontsize=10.2)
ax.set_title("B. Donor-Level Within-Section Gradient vs. Disease Progression\n(Exposing Single-Donor Pseudoreplication)", fontsize=11.5, fontweight='bold', pad=10)
ax.grid(True, alpha=0.25)

leg_7b = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#C0392B', markeredgecolor='#111', markersize=7.5, label='Dementia (APOE4−)'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='#C0392B', markeredgecolor='#111', markersize=7.5, label='Dementia (APOE4+)'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='#C0392B', markeredgecolor='#111', markersize=12, label='Mega-Donor H20.33.004 (Dem, APOE4+)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2980B9', markeredgecolor='#111', markersize=7.5, label='No Dementia (APOE4−)'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='#2980B9', markeredgecolor='#111', markersize=7.5, label='No Dementia (APOE4+)'),
]
ax.legend(handles=leg_7b, fontsize=7.6, loc='lower right', frameon=True, facecolor='white', framealpha=0.95)

stat_7b = (
    f"Spearman ρ = {rho_cps:+.3f}, p = {p_cps:.4f} (n = 27)\n"
    f"Donor MWU: Dementia p = {p_cog:.3f} (NS)\n"
    f"Donor MWU: APOE4+   p = {p_apoe:.3f} (NS)"
)
ax.text(0.03, 0.96, stat_7b, transform=ax.transAxes, fontsize=8.2, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#F8F9FA', edgecolor='#666666', alpha=0.95))

# Panel C: Stacked Bar of Cell-Type Composition (0-25, 25-50, 50-150 um)
ax = axes[1, 0]
x_z = np.arange(len(niche_df))
w_s = 0.52
cats = [
    ('Neurons', '#2980B9', 'Neurons (Glut + GABA)'),
    ('Microglia-PVM', '#C0392B', 'Microglia-PVM'),
    ('Oligodendrocyte', '#95A5A6', 'Oligodendrocytes'),
    ('Astrocyte', '#27AE60', 'Astrocytes'),
    ('Other', '#F1C40F', 'OPC / Vascular / Other'),
]
bottom = np.zeros(len(niche_df))
for col_n, col_hex, lbl_n in cats:
    vals_c = niche_df[col_n].values
    ax.bar(x_z, vals_c, bottom=bottom, width=w_s, color=col_hex, edgecolor='#222222', label=lbl_n, alpha=0.90)
    for xi, (v, b) in enumerate(zip(vals_c, bottom)):
        if v >= 6.5:
            ax.text(xi, b + v/2, f"{v:.1f}%", ha='center', va='center', fontsize=8.6, fontweight='bold',
                    color='white' if col_n in ('Neurons', 'Microglia-PVM') else '#111111')
    bottom += vals_c

neu_dep = (1.0 - niche_df.loc[0, 'Neurons'] / niche_df.loc[2, 'Neurons']) * 100.0
mg_enr = (niche_df.loc[0, 'Microglia-PVM'] / niche_df.loc[2, 'Microglia-PVM'] - 1.0) * 100.0
ax.set_xticks(x_z)
ax.set_xticklabels(niche_df['Zone'], fontsize=9.8, fontweight='bold')
ax.set_ylim(0, 118)
ax.set_ylabel("Cell-Type Composition (%)", fontsize=10.5)
ax.set_title("C. Strictly Within-Section Multicellular Niche Composition\n(1.89M Cells in X_spatial_tiled, Global Top-500 Anchors)", fontsize=11.5, fontweight='bold', pad=10)
ax.legend(fontsize=8.0, loc='upper right', ncol=2, frameon=True, facecolor='white', framealpha=0.95)
ax.text(0.02, 0.96, f"Inner Halo (<25 µm vs. Distal):\n• −{neu_dep:.1f}% Neuronal Depletion\n• +{mg_enr:.1f}% Microglial Enrichment",
        transform=ax.transAxes, fontsize=8.3, fontweight='bold', va='top', ha='left', color='#1B4F72',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#EBF5FB', edgecolor='#2980B9', alpha=0.95))

# Panel D: 2D SHAP Dependence of LRRK1
ax = axes[1, 1]
sc_7d = ax.scatter(dist_7d, shap_lrrk1, c=np.clip(expr_lrrk1_7d, 0, np.percentile(expr_lrrk1_7d, 98)),
                   cmap='viridis', s=12, alpha=0.65, edgecolors='none')
cb = fig.colorbar(sc_7d, ax=ax, pad=0.02)
cb.set_label('LRRK1 Expression (Volume-Scaled Spots)', fontsize=9.5)
ax.axhline(0, color='#555555', ls='--', lw=1.2)
ax.axvline(25, color='#C0392B', ls=':', lw=2.0, label='25 µm Proximal Zone')
ax.set_xlim(0, 140)
ax.set_xlabel("Within-Section Proxy Distance (µm)", fontsize=10.5)
ax.set_ylabel("SHAP Impact of LRRK1 on Activation Log-Odds", fontsize=10.5)
ax.set_title("D. 2D SHAP Dependence of LRRK1 Across Distance\n(Horizontal Stratification Confirms Expression-Driven Impact)", fontsize=11.5, fontweight='bold', pad=10)
ax.legend(fontsize=8.6, loc='upper right', frameon=True, facecolor='white', framealpha=0.95)
ax.grid(True, alpha=0.25)

fig.tight_layout(h_pad=3.0, w_pad=2.5)
save_fig(fig, 'Figure_7_Deconvolution_Pseudoreplication_Niche_SHAP.png')

# ==============================================================================
# FIGURE 8: Monte Carlo Power (True vs Proxy), Bias, 4-Method Fits, Synthetic MAE
# ==============================================================================
print("Generating Figure 8...", flush=True)
fig = plt.figure(figsize=(16.0, 11.8), dpi=300)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.26)

# Panel A split into A1 (True Centroids Power) and A2 (Within-Section Proxy Power)
gs_a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0], wspace=0.22)
ax_a1 = fig.add_subplot(gs_a[0, 0])
ax_a2 = fig.add_subplot(gs_a[0, 1])

sec_list = [1, 10, 30, 69]
sec_cols = ['#95A5A6', '#3498DB', '#E67E22', '#C0392B']
for ns, col in zip(sec_list, sec_cols):
    sub_s = sim_df[sim_df['n_sections'] == ns].sort_values('cells_per_section')
    ax_a1.plot(sub_s['cells_per_section'], sub_s['true_plaque_power_pct'], 'o-', color=col, lw=2.2, markersize=5.5, label=f'{ns} sec')
    ax_a2.plot(sub_s['cells_per_section'], sub_s['within_proxy_power_pct'], 's--', color=col, lw=2.2, markersize=5.5, label=f'{ns} sec')

for a_sub, ttl in [(ax_a1, 'A1. True Plaque Centroids\nDetection Power (%)'), (ax_a2, 'A2. Within-Section Proxies\nPower Collapse (0–22.5% @50)')]:
    a_sub.set_xscale('log')
    a_sub.set_xticks([50, 100, 500, 1500])
    a_sub.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    a_sub.set_ylim(-5, 108)
    a_sub.set_xlabel("Cells / Section (log)", fontsize=9.8)
    a_sub.set_title(ttl, fontsize=10.5, fontweight='bold', pad=8)
    a_sub.grid(True, alpha=0.25)
    a_sub.legend(fontsize=7.8, loc='lower right', frameon=True, facecolor='white', framealpha=0.92)
ax_a1.set_ylabel("Power (% Detecting True 30 µm Knot)", fontsize=10.0)

# Panel B: Estimated Knot vs Number of Sections (True vs Within-Proxy vs Confounded Pooled)
ax_b = fig.add_subplot(gs[0, 1])
s100 = sim_df[sim_df['cells_per_section'] == 100].sort_values('n_sections')
ax_b.axhline(30.0, color='#111111', ls=':', lw=1.6, label='Ground-Truth Physical Boundary (30.0 µm)')
ax_b.errorbar(s100['n_sections'], s100['true_plaque_knot_mean'], yerr=s100['true_plaque_knot_sd'], fmt='o-', color='#117A65', lw=2.4, capsize=4, label='True Plaque Centroids (~30 µm flat)')
ax_b.errorbar(s100['n_sections'], s100['within_proxy_knot_mean'], yerr=s100['within_proxy_knot_sd'], fmt='s--', color='#E67E22', lw=2.2, capsize=4, label='Within-Section Proxies (wide scatter)')
ax_b.plot(s100['n_sections'], s100['confounded_pooled_knot_mean'], 'D-', color='#C0392B', lw=2.4, markersize=6.5, label='Confounded Pooled Proxies (inward pull → 11.5 µm)')
ax_b.set_xticks([1, 10, 30, 69])
ax_b.set_ylim(5, 52)
ax_b.set_xlabel("Number of Pooled Cortical Sections", fontsize=10.5)
ax_b.set_ylabel("Estimated Decay Breakpoint (µm ± SD)", fontsize=10.5)
ax_b.set_title("B. Monte Carlo Estimator Bias Across Section Count (100 cells/sec)\nExplaining Pooled Inward Pull (11.5–15.8 µm)", fontsize=11.5, fontweight='bold', pad=10)
ax_b.legend(fontsize=8.4, loc='upper right', frameon=True, facecolor='white', framealpha=0.95)
ax_b.grid(True, alpha=0.25)

# Panel C: Four Method Fits on Empirical CosMx Cd74+Ctss (CP1K) Curve
ax_c = fig.add_subplot(gs[1, 0])
ax_c.plot(bx_all, prof_c['norm_cd74'].values, 'o-', color='#333333', lw=1.8, markersize=4.5, alpha=0.75, label='CosMx CP1K Cd74 + Ctss')
ax_c.plot(bx_all, m1_fit.predict(X_fit1_all), '-', color='#C0392B', lw=2.5, label='Segmented Hinge (33.0 µm)')
ax_c.axvline(33.0, color='#C0392B', ls='-', lw=2.0)
ax_c.axvline(32.8, color='#2980B9', ls='--', lw=2.0, label='GAM Cubic Spline 90% Decay (32.8 µm)')
ax_c.axvline(28.0, color='#E67E22', ls='-.', lw=2.0, label='PELT Unconstrained 2-Seg (28.0 µm)')
ax_c.axvline(18.0, color='#8E44AD', ls=':', lw=2.2, label='Binseg Binary Step (18.0 µm)')
ax_c.set_xlim(0, 100)
ax_c.set_xlabel("Distance to Physical Aβ Plaque Centroid (µm)", fontsize=10.5)
ax_c.set_ylabel("Normalized Cd74 + Ctss (CP1K)", fontsize=10.5)
ax_c.set_title("C. 4-Method Breakpoint Comparison on Empirical CosMx Curve", fontsize=11.5, fontweight='bold', pad=10)
ax_c.legend(fontsize=8.5, loc='upper right', frameon=True, facecolor='white', framealpha=0.95)
ax_c.grid(True, alpha=0.25)

# Panel D: Bar Chart of Synthetic MAE
ax_d = fig.add_subplot(gs[1, 1])
m_names = ['Segmented\nHinge', 'GAM Cubic\nSpline', 'PELT\n(2-Seg)', 'Binseg\n(Step)']
m_maes = [0.94, 1.27, 3.28, 12.45]
m_cols = ['#C0392B', '#2980B9', '#E67E22', '#8E44AD']
bars_8d = ax_d.bar(m_names, m_maes, color=m_cols, edgecolor='#222222', width=0.55, alpha=0.90)
for b, v in zip(bars_8d, m_maes):
    ax_d.text(b.get_x() + b.get_width()/2, v + 0.3, f"{v:.2f} µm", ha='center', va='bottom', fontsize=9.5, fontweight='bold')
ax_d.set_ylim(0, 14.5)
ax_d.set_ylabel("Mean Absolute Error vs. True 30 µm Knot (µm)", fontsize=10.5)
ax_d.set_title("D. Synthetic Benchmark Accuracy (40 Replicates, True Knot = 30 µm)", fontsize=11.5, fontweight='bold', pad=10)
ax_d.grid(True, alpha=0.25, axis='y')

save_fig(fig, 'Figure_8_MonteCarlo_and_Method_Benchmark.png')

# ==============================================================================
# SUPP FIG S2: Pooled Spatial Architecture & Pooled Activation Curve
# ==============================================================================
print("Generating Supplementary Figure S2...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), dpi=300)
sub_s2 = rng_s.choice(len(df_mg), size=25000, replace=False)
ax = axes[0]
sc_s2 = ax.scatter(df_mg['X_Pooled'].values[sub_s2], df_mg['Y_Pooled'].values[sub_s2],
                   c=np.clip(act_score[sub_s2], 0, 120), cmap='magma', s=2.5, alpha=0.5, edgecolors='none')
cb2 = fig.colorbar(sc_s2, ax=ax, pad=0.02)
cb2.set_label('CD74 + CTSS Score', fontsize=9.5)
ax.set_xlabel("Pooled X Coordinate (µm, obsm['spatial'])", fontsize=10.2)
ax.set_ylabel("Pooled Y Coordinate (µm, obsm['spatial'])", fontsize=10.2)
ax.set_title("A. Overlaid 69-Section Coordinate Field (obsm['spatial'])", fontsize=11.5, fontweight='bold', pad=10)

ax2 = axes[1]
d_p140 = df_mg[(df_mg['Dist_Pooled'] > 0) & (df_mg['Dist_Pooled'] <= 140)].copy()
d_p140['Bin'] = pd.cut(d_p140['Dist_Pooled'], bins=np.arange(0, 145, 5), labels=np.arange(2.5, 142.5, 5)).astype(float)
p_s2 = d_p140.groupby('Bin', observed=False)['Event_Q90'].mean() * 100.0
ax2.plot(p_s2.index.values, p_s2.values, 'o-', color='#C0392B', lw=2.2, markersize=5, label='Pooled Extreme Activation Rate (≥Q90 %)')
ax2.axvline(24.0, color='#1B4F72', ls='--', lw=2.0, label='Apparent Pooled Knot (~24–25 µm)')
ax2.set_xlabel("Pooled Distance to Top-500 Anchors (µm)", fontsize=10.2)
ax2.set_ylabel("Extreme Activation Prevalence (%)", fontsize=10.2)
ax2.set_title("B. Pooled-Coordinate Activation Curve (Artifactual Aggregation)", fontsize=11.5, fontweight='bold', pad=10)
ax2.legend(fontsize=8.8, loc='upper right', frameon=True, facecolor='white')
ax2.grid(True, alpha=0.25)

fig.text(0.5, -0.02, "Footnote: Pooled obsm['spatial'] coordinates; not for within-section inference.",
         ha='center', fontsize=9.0, style='italic', fontweight='bold', color='#922B21')
fig.tight_layout(w_pad=2.5)
save_fig(fig, 'SuppFig_S2_Pooled_Spatial_Architecture_and_Curve.png')

# ==============================================================================
# SUPP FIG S3: Moran's I Correlogram (A) & 100 Toroidal Shifts (B)
# ==============================================================================
print("Generating Supplementary Figure S3...", flush=True)
moran_df = pd.DataFrame(sum_base['analysis_3_spatial_autocorrelation']['moran_correlogram'])
toro = sum_base['analysis_3_spatial_autocorrelation']['toroidal_shift_null_100']

fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), dpi=300)
ax = axes[0]
ax.plot(moran_df['mid_um'], moran_df['moran_I'], 'o-', color='#1B4F72', lw=2.4, markersize=7)
ax.fill_between(moran_df['mid_um'], 0, moran_df['moran_I'], color='#AED6F1', alpha=0.35)
peak_r = moran_df.loc[moran_df['moran_I'].idxmax()]
ax.annotate(f"Peak at 50–75 µm\nI = {peak_r['moran_I']:.4f}, Z = {peak_r['z_score']:.2f}",
            xy=(peak_r['mid_um'], peak_r['moran_I']), xytext=(70, 0.12),
            fontsize=9.0, fontweight='bold', color='#922B21',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#FDEDEC', edgecolor='#C0392B'),
            arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.5))
ax.set_xlabel("Spatial Lag Ring Midpoint (µm)", fontsize=10.5)
ax.set_ylabel("Moran's I Spatial Autocorrelation", fontsize=10.5)
ax.set_title("A. Spatial Autocorrelation Correlogram (Moran's I)", fontsize=11.5, fontweight='bold', pad=10)
ax.set_ylim(0, 0.19)
ax.grid(True, alpha=0.25)

ax2 = axes[1]
xt = np.linspace(-3.5, 3.5, 400)
yt = norm.pdf(xt, toro['mean_elev25_pct'], toro['sd_elev25_pct'])
ax2.fill_between(xt, 0, yt, color='#BDC3C7', alpha=0.55, label=f"100 Toroidal Shifts\n(Mean = {toro['mean_elev25_pct']:+.3f}% ± {toro['sd_elev25_pct']:.3f}% SD)")
ax2.plot(xt, yt, color='#555555', lw=2.0)
ax2.axvline(toro['observed_elev25_pct'], color='#C0392B', ls='--', lw=2.5,
            label=f"Observed Proximal Elevation ({toro['observed_elev25_pct']:+.2f}%)\nEmpirical p = {toro['empirical_p_value']:.4f}")
ax2.set_xlabel("Proximal (<25 µm) Elevation in Extreme Activation (%)", fontsize=10.5)
ax2.set_ylabel("Null Density", fontsize=10.5)
ax2.set_title("B. 100-Iteration Toroidal Spatial Shift Null Control", fontsize=11.5, fontweight='bold', pad=10)
ax2.legend(fontsize=8.6, loc='upper left', frameon=True, facecolor='white')
ax2.grid(True, alpha=0.25)

fig.tight_layout(w_pad=2.5)
save_fig(fig, 'SuppFig_S3_Spatial_Autocorrelation_MoransI.png')

# ==============================================================================
# SUPP FIG S4: Alternative Activation Definitions
# ==============================================================================
print("Generating Supplementary Figure S4...", flush=True)
alt = sum_base['analysis_4_alternative_scores']
alt_names = ['Baseline\n(CD74+CTSS)', 'CD74\nAlone', 'CTSS\nAlone', '4-Gene\nCascade', 'Broad 6-Gene\nSignature']
alt_keys = ['Baseline (CD74 + CTSS)', 'CD74 Alone', 'CTSS Alone', 'Secondary Cascade (LRRK1 + L3MBTL4 + PDE4B + UNC5B)', 'Broad 6-Gene Myeloid Signature']
alt_knots = [alt[k]['knot_vs_baseline_anchors_um'] for k in alt_keys]
alt_lrs = [alt[k]['lr_stat_continuous'] for k in alt_keys]

fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4), dpi=300)
ax = axes[0]
cols_s4 = ['#C0392B', '#E74C3C', '#2980B9', '#E67E22', '#117A65']
b_s4 = ax.bar(alt_names, alt_knots, color=cols_s4, edgecolor='#222222', width=0.58, alpha=0.90)
for b, v in zip(b_s4, alt_knots):
    ax.text(b.get_x() + b.get_width()/2, v + 0.8, f"{v} µm", ha='center', va='bottom', fontsize=9.2, fontweight='bold')
ax.axhline(25, color='#333333', ls='--', lw=1.4, alpha=0.7)
ax.set_ylim(0, 52)
ax.set_ylabel("Optimal Pooled Breakpoint (µm)", fontsize=10.5)
ax.set_title("A. Breakpoint Sensitivity Across Activation Definitions", fontsize=11.5, fontweight='bold', pad=10)
ax.grid(True, alpha=0.25, axis='y')

ax2 = axes[1]
b_s4b = ax2.bar(alt_names, alt_lrs, color=cols_s4, edgecolor='#222222', width=0.58, alpha=0.90)
for b, v in zip(b_s4b, alt_lrs):
    ax2.text(b.get_x() + b.get_width()/2, v + 0.18, f"{v:.2f}", ha='center', va='bottom', fontsize=9.2, fontweight='bold')
ax2.set_ylim(0, 10.5)
ax2.set_ylabel("Piecewise vs. Linear Likelihood-Ratio Statistic", fontsize=10.5)
ax2.set_title("B. Nonlinearity Strength Across Marker Definitions", fontsize=11.5, fontweight='bold', pad=10)
ax2.grid(True, alpha=0.25, axis='y')

fig.text(0.5, -0.02, "Note: CD74 drives the pooled signal (24 µm); CTSS alone shifts to 43 µm (computed in pooled obsm['spatial'] coordinates).",
         ha='center', fontsize=9.0, style='italic', fontweight='bold', color='#1B4F72')
fig.tight_layout(w_pad=2.5)
save_fig(fig, 'SuppFig_S4_Alternative_Activation_Definitions.png')

# ==============================================================================
# SUPP FIG S5: Diffusion Pseudotime vs. Physical Distance
# ==============================================================================
print("Generating Supplementary Figure S5...", flush=True)
# Compute real diffusion pseudotime on a representative 8,000-cell subsample of Microglia-PVM
sub_dpt_idx = rng_s.choice(len(mg), size=8000, replace=False)
mg_sub = mg[sub_dpt_idx].copy()
sc.pp.pca(mg_sub, n_comps=15, random_state=42)
sc.pp.neighbors(mg_sub, n_neighbors=15, random_state=42)
sc.tl.diffmap(mg_sub, n_comps=15)
mg_sub.uns['iroot'] = int(np.argmin(act_score[sub_dpt_idx]))
sc.tl.dpt(mg_sub)
dpt_vals = mg_sub.obs['dpt_pseudotime'].values
dist_dpt = df_mg['Dist_WithinSec'].values[sub_dpt_idx]
valid_dpt = np.isfinite(dpt_vals) & np.isfinite(dist_dpt) & (dist_dpt <= 2500)
r_dpt, p_dpt = spearmanr(dpt_vals[valid_dpt], dist_dpt[valid_dpt])

fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=300)
hb = ax.hexbin(dist_dpt[valid_dpt], dpt_vals[valid_dpt], gridsize=45, cmap='viridis', mincnt=1)
cb5 = fig.colorbar(hb, ax=ax, pad=0.02)
cb5.set_label('Cell Count per Hexbin', fontsize=9.8)
ax.axvspan(33, 36, color='#C0392B', alpha=0.45, label='CosMx Physical Plaque Boundary (33–36 µm)')
ax.set_xlabel("Within-Section Physical Distance to Nearest Anchor (µm)", fontsize=10.5)
ax.set_ylabel("Diffusion Pseudotime (DPT)", fontsize=10.5)
ax.set_title("Supplementary Figure S5. Transcriptomic Diffusion Pseudotime vs. Physical Distance\n(SEA-AD Microglia-PVM Subsample, n = 8,000)", fontsize=11.5, fontweight='bold', pad=10)
ax.text(0.97, 0.94, f"Pseudotime smears across >2,000 µm;\ndoes not recover the 33–36 µm physical boundary\n(Spearman ρ = {r_dpt:+.3f}, p = {p_dpt:.3e})",
        transform=ax.transAxes, fontsize=8.8, fontweight='bold', ha='right', va='top', color='#922B21',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FDEDEC', edgecolor='#C0392B', alpha=0.95))
ax.legend(fontsize=8.8, loc='lower right', frameon=True, facecolor='white')
ax.grid(True, alpha=0.25)
save_fig(fig, 'SuppFig_S5_Diffusion_Pseudotime_vs_Distance.png')

# ==============================================================================
# SUPP FIG S6: Anchor-Count Sensitivity (A) & 138-Gene Tier-1 Spillover Histogram (B)
# ==============================================================================
print("Generating Supplementary Figure S6...", flush=True)
anc_sens = sum_base['analysis_1_negative_controls']['activation_anchors']
anc_k = [250, 500, 750, 1000]
anc_elev = [anc_sens[str(k)]['elev25_pct'] for k in anc_k]
anc_knots = [anc_sens[str(k)]['knot_q90'] for k in anc_k]

fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), dpi=300)
ax = axes[0]
b_s6 = ax.bar([str(k) for k in anc_k], anc_elev, color='#2E86C1', edgecolor='#1B4F72', width=0.52, alpha=0.88)
for b, e, kn in zip(b_s6, anc_elev, anc_knots):
    ax.text(b.get_x() + b.get_width()/2, e + 0.04, f"+{e:.2f}%\n(knot={kn}µm)", ha='center', va='bottom', fontsize=8.6, fontweight='bold')
ax.set_ylim(0, 1.95)
ax.set_xlabel("Number of Top-Activated Anchor Cells", fontsize=10.5)
ax.set_ylabel("Proximal (<25 µm) Extreme Activation Elevation (%)", fontsize=10.2)
ax.set_title("A. Anchor-Count Sensitivity Analysis", fontsize=11.5, fontweight='bold', pad=10)
ax.grid(True, alpha=0.25, axis='y')
ax.text(0.5, -0.18, "Footnote: Computed in pooled obsm['spatial'] coordinates.", transform=ax.transAxes,
        ha='center', fontsize=8.6, style='italic', color='#922B21')

ax2 = axes[1]
t1_ratios = t_s2[~t_s2['Gene'].isin(['CD74', 'CTSS'])]['Max_Ratio'].dropna().values
n_flagged = int(np.sum(t1_ratios >= 1.5))
ax2.hist(np.clip(t1_ratios, 0, 5.0), bins=30, color='#E67E22', edgecolor='#222222', alpha=0.82)
ax2.axvline(1.5, color='#C0392B', ls='--', lw=2.4, label=f'Tier-1 Spillover Cutoff (1.5×)\n{n_flagged}/138 genes flagged ({100*n_flagged/138:.1f}%)')
ax2.set_xlabel("Maximum Broad-Class Neighbor Spillover Ratio (<10 µm / >20 µm)", fontsize=10.2)
ax2.set_ylabel("Number of MERFISH Panel Genes", fontsize=10.5)
ax2.set_title("B. Tier-1 Neighbor Spillover Ratio Distribution (138 Genes)", fontsize=11.5, fontweight='bold', pad=10)
ax2.legend(fontsize=8.8, loc='upper right', frameon=True, facecolor='white')
ax2.grid(True, alpha=0.25, axis='y')

fig.tight_layout(w_pad=2.5)
save_fig(fig, 'SuppFig_S6_Anchor_Sensitivity_and_Spillover_Dist.png')

# ==============================================================================
# NEW SUPP FIG S7: CosMx Raw vs. CP1K-Normalized Breakpoint Shift (31->33, 31->36)
# ==============================================================================
print("Generating NEW Supplementary Figure S7...", flush=True)
# Fit raw curves over 0-140 um at their exact verified knots (31 um)
X_raw1 = np.column_stack([np.ones_like(bx_all), np.minimum(bx_all, 31), np.maximum(0, bx_all - 31)])
pred_raw1 = sm.OLS(prof_c['raw_cd74'].values, X_raw1).fit().predict(X_raw1)
pred_raw2 = sm.OLS(prof_c['raw_dam10'].values, X_raw1).fit().predict(X_raw1)

fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.8), dpi=300)

# Panel A: Cd74+Ctss Raw (31 um) vs CP1K (33 um)
ax = axes[0]
ax.plot(bx_all, prof_c['norm_cd74'].values, 'o-', color='#C0392B', lw=2.4, markersize=5, label='CP1K-Normalized Cd74+Ctss (Knot = 33 µm, CI 31–37)')
ax.axvline(33.0, color='#C0392B', ls='-', lw=2.2)
ax_t1 = ax.twinx()
ax_t1.plot(bx_all, prof_c['raw_cd74'].values, 's--', color='#5D6D7E', lw=1.9, markersize=4.5, alpha=0.75, label='Raw Unnormalized Cd74+Ctss (Knot = 31 µm, CI 29–35)')
ax_t1.axvline(31.0, color='#5D6D7E', ls='--', lw=1.8)
ax.set_xlim(0, 100)
ax.set_xlabel("Distance to True Anti-Aβ Plaque Centroid (µm)", fontsize=10.5)
ax.set_ylabel("CP1K-Normalized Cd74 + Ctss", color='#C0392B', fontsize=10.5)
ax_t1.set_ylabel("Raw Cd74 + Ctss Counts", color='#5D6D7E', fontsize=10.5)
ax.set_title("A. Cd74 + Ctss Normalization Shift: 31 µm (Raw) → 33 µm (CP1K)", fontsize=11.5, fontweight='bold', pad=10)
ax.grid(True, alpha=0.25)
h_a1, l_a1 = ax.get_legend_handles_labels()
h_a2, l_a2 = ax_t1.get_legend_handles_labels()
ax.legend(h_a1 + h_a2, l_a1 + l_a2, fontsize=8.3, loc='upper right', frameon=True, facecolor='white')

# Panel B: 10-Gene DAM Raw (31 um) vs CP1K (36 um)
ax2 = axes[1]
ax2.plot(bx_all, prof_c['norm_dam10'].values, 'o-', color='#D35400', lw=2.4, markersize=5, label='CP1K-Normalized 10-Gene DAM (Knot = 36 µm, CI 34–40)')
ax2.axvline(36.0, color='#D35400', ls='-', lw=2.2)
ax_t2 = ax2.twinx()
ax_t2.plot(bx_all, prof_c['raw_dam10'].values, 's--', color='#5D6D7E', lw=1.9, markersize=4.5, alpha=0.75, label='Raw Unnormalized 10-Gene DAM (Knot = 31 µm, CI 29–38)')
ax_t2.axvline(31.0, color='#5D6D7E', ls='--', lw=1.8)
ax2.set_xlim(0, 100)
ax2.set_xlabel("Distance to True Anti-Aβ Plaque Centroid (µm)", fontsize=10.5)
ax2.set_ylabel("CP1K-Normalized 10-Gene DAM", color='#D35400', fontsize=10.5)
ax_t2.set_ylabel("Raw 10-Gene DAM Counts", color='#5D6D7E', fontsize=10.5)
ax2.set_title("B. 10-Gene DAM Normalization Shift: 31 µm (Raw) → 36 µm (CP1K)", fontsize=11.5, fontweight='bold', pad=10)
ax2.grid(True, alpha=0.25)
h_b1, l_b1 = ax2.get_legend_handles_labels()
h_b2, l_b2 = ax_t2.get_legend_handles_labels()
ax2.legend(h_b1 + h_b2, l_b1 + l_b2, fontsize=8.3, loc='upper right', frameon=True, facecolor='white')

fig.tight_layout(w_pad=3.0)
save_fig(fig, 'SuppFig_S7_CosMx_Raw_vs_CP1K_Comparison.png')

# ==============================================================================
# NEW SUPP FIG S8: Tier-2 Fine-Subclass Expression Heatmap (24 Subclasses x 8 Genes)
# ==============================================================================
print("Generating NEW Supplementary Figure S8...", flush=True)
# Order subclasses to highlight Microglia-PVM, Sst Chodl, Endothelial, OPC at the top
priority_sub = ['Microglia-PVM', 'Sst Chodl', 'Sst', 'Endothelial', 'VLMC', 'OPC', 'Oligodendrocyte', 'Astrocyte']
other_sub = sorted([s for s in sub_means_s8.index if s not in priority_sub])
ordered_sub = [s for s in priority_sub if s in sub_means_s8.index] + other_sub
mat_raw = sub_means_s8.loc[ordered_sub, s8_genes]
# Column-wise min-max / max-ratio scaling (fraction of each gene's peak subclass expression) so every gene's true host cell type jumps out!
mat_norm = mat_raw.div(mat_raw.max(axis=0), axis=1)

fig, ax = plt.subplots(figsize=(11.5, 9.5), dpi=300)
im = ax.imshow(mat_norm.values, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1.0)
cb8 = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
cb8.set_label('Relative Mean Expression (Fraction of Peak Subclass)', fontsize=10.0)

ax.set_xticks(np.arange(len(s8_genes)))
ax.set_xticklabels(s8_genes, fontsize=10.5, fontweight='bold')
ax.set_yticks(np.arange(len(ordered_sub)))
ax.set_yticklabels(ordered_sub, fontsize=9.2)

# Annotate actual mean spot counts inside cells
for i in range(len(ordered_sub)):
    for j in range(len(s8_genes)):
        raw_v = mat_raw.values[i, j]
        rel_v = mat_norm.values[i, j]
        txt_col = 'white' if rel_v >= 0.60 else '#111111'
        ax.text(j, i, f"{raw_v:.2f}", ha='center', va='center', fontsize=7.6,
                fontweight='bold' if rel_v >= 0.60 else 'normal', color=txt_col)

# Vertical divider separating Tier-2 Spillover Genes (left 4) from Pure Intrinsic Myeloid Drivers (right 4)
ax.axvline(3.5, color='#1B4F72', lw=2.8, ls='-')
ax.text(1.5, -1.05, "Tier-2 Rare-Subclass Spillover Genes", ha='center', va='center', fontsize=10.2, fontweight='bold', color='#D35400')
ax.text(5.5, -1.05, "Pure Intrinsic Myeloid Drivers (<1.5× everywhere)", ha='center', va='center', fontsize=10.2, fontweight='bold', color='#922B21')

ax.set_title("Supplementary Figure S8. Fine-Grained Expression Across All 24 SEA-AD Subclasses (1,887,729 Cells)\nVisual Proof of Tier-2 Rare-Subclass Spillover (TACR1/PLD5 in Sst Chodl; RFX3 in Endothelial; TMEM255A in OPC)",
             fontsize=11.5, fontweight='bold', pad=28)
save_fig(fig, 'SuppFig_S8_Tier2_Subclass_Expression_Heatmap.png')

print("\n=== ALL 16 CONSISTENTLY NAMED PUBLICATION FIGURES GENERATED SUCCESSFULLY! ===", flush=True)
