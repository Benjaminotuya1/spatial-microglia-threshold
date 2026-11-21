# Five Methodological Pitfalls in Imaging Spatial Transcriptomics of Alzheimer's Disease

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Official reproducible code and analysis repository for the manuscript:

> **Five Methodological Pitfalls in Imaging Spatial Transcriptomics of Alzheimer's Disease: A Multi-Platform Audit of Transcriptomic Plaque Proxies, Two-Tier Spillover, and Donor-Level Generalization**  
> Benjamin Goodluck Otuya  
> Department of Human Physiology, Ahmadu Bello University, Zaria, Nigeria  
> Contact: otuyabenjamin2000@gmail.com

---

## Overview & Key Findings

This repository provides a complete, reproducible multi-platform audit of spatial microglial activation across **1,887,729 human middle temporal gyrus (MTG) MERFISH cells** (`195,620` `Microglia-PVM` across `27` donors and `69` cortical sections; SEA-AD) and **43,117 NanoString CosMx cells** (`11,434` microglia co-registered with **`402` physical anti-Aβ `MOAB-2` immunofluorescence plaques** across `4` coronal hemisphere sections from 18-month *App*$^{NL-G-F}$ mice; NCBI GEO `GSE263791`, Mallach et al., *Cell Reports* 2024).

### The Five Interlocking Methodological Pitfalls Audited
1. **Pitfall 1 — Matrix Normalization Metadata Mismatch:** SEA-AD `adata.uns['X_normalization']` labels the matrix as `ln(Spots per 10K +1)`, yet direct inspection proves `adata.X` stores **linear volume-scaled spot counts** (`max = 483.0`, matching `obs['Number of spots']` with $r = 1.000000$, `max_abs_diff = 0.0`).
2. **Pitfall 2 — Two-Tier Neighbor-Proximity Transcript Spillover:**
   - *Tier-1 Broad-Class Audit (`<10 µm` vs. `>20 µm` in `X_spatial_tiled`):* Flags **`83/138` genes (`60.1%`)** at $\ge 1.5	imes$ (e.g., `RBFOX3` `2.48×` Neuron, `MOG` `3.32×` Oligo, `CD22` `3.89×` Oligo).
   - *Tier-2 Fine-Grained 24-Subclass Audit:* Exposes rare-subclass contamination invisible to pooled-neuron screening — **`TACR1` (`2.32×` near `Sst Chodl` interneurons)**, **`PLD5` (`2.17×` `Sst Chodl`)**, **`TMEM255A` (`1.75×` `OPC`)**, and **`RFX3` (`1.60×` `Endothelial`)**.
   - *Spillover-Resistant Intrinsic Myeloid Core (`<1.5×` across all 24 subclasses):* **`LRRK1` (`0.87×` max)**, **`L3MBTL4` (`1.41×`)**, **`PDE4B` (`1.42×`)**, and **`UNC5B` (`1.32×`)**.
3. **Pitfall 3 — Pooled-Section Coordinate Confounding & Single-Section Proxy Collapse:** Overlaying `69` sections in `obsm['spatial']` pulls apparent breakpoints inward (`~24–25 µm`). Strictly within-section proxy knots scatter widely across `57` evaluable sections (**Median `56 µm`, IQR `33–71 µm`**), and Monte Carlo simulations confirm power collapses to **`0–22.5%`** at `50` cells/section.
4. **Pitfall 4 — Changepoint Estimator Bias & Capture-Rate Normalization:**
   - In CosMx (`402` physical plaques, $d = 0$ excluded), total-transcript normalization (**CP1K**, counts per 1,000 transcripts) yields a reproducible physical peri-plaque decay radius of **`33.0 µm` (95% plaque-cluster CI: `31.0–37.0 µm`)** for `Cd74 + Ctss` and **`36.0 µm` (95% CI: `34.0–40.0 µm`)** for `10-Gene DAM` (raw-count shift: `31 → 33 µm` and `31 → 36 µm`), while correcting the unnormalized `P2ry12 + Tmem119` suppression artifact (`2.86 CP1K` at `0–5 µm` vs. `1.97 CP1K` at `50–140 µm`).
   - Across `4` changepoint estimators on synthetic `30 µm` ground truth, **Two-Piece Segmented Hinge Regression** achieves the lowest MAE (**`0.94 µm`**) vs. `GAM` (`1.27 µm`), `PELT` (`3.28 µm`), and `Binseg` (`12.45 µm`).
5. **Pitfall 5 — Single-Donor Pseudoreplication & Donor-Level Generalization:**
   - One mega-donor (**`H20.33.004`**, `37,874` microglia, `397` proximal cells) supplies **`65.5%`** of Dementia and **`82.0%`** of `APOE4+` proximal cells. Patient-level two-sided Mann-Whitney U tests (`n = 27`) are non-significant (`Dementia p = 0.114`, `APOE4 p = 0.289`), whereas continuous neuropathological progression (**`CPS` vs. donor $\Delta	ext{Score}$**) remains significant (**Spearman $ho = +0.397, p = 0.0404$**), alongside strictly within-section **`−39.4%` neuronal depletion** and **`+58.2%` microglial enrichment** inside `<25 µm`.
   - 5-fold donor-level `GroupKFold` XGBoost achieves **`0.8910 ± 0.0113`** cell-weighted ROC-AUC (**27-donor median `0.8821`**, IQR `0.8498–0.8970`; `Genes Only 138 = 0.8912`, `Genes + WithinSec Dist = 0.8917`, `54 Decontaminated Genes = 0.8810`, `Distance Only = 0.5568`).

---

## Repository Contents

### Master Executed Notebook
- `02_Spatial_Probability_Model_Complete_Reanalysis.ipynb` — End-to-end executed Jupyter notebook (Parts I–VI) covering baseline replication, all 5 methodological pitfall audits, donor-level CV ablation, CosMx CP1K validation, and simulation benchmarking.

### Reproducible Python Scripts
- `download_data.py` — Downloads `SEAAD_MTG_MERFISH.2024-12-11.h5ad` from the Allen Institute AWS Open Data Registry (`s3://sea-ad-spatial-transcriptomics/`).
- `download_cosmx_plaque.py` — Downloads `GSM8199188` and `GSM8199189` (`GSE263791`) from NCBI GEO.
- `run_full_reanalysis.py` — Baseline GMM thresholding, XGBoost classifier, SHAP, and Reviewer Analyses 1–6.
- `run_advanced_upgrades.py` — Myeloid `Supertype` deconvolution, clinical/APOE4 stratification, and multicellular niche profiling.
- `run_critical_audit_and_decontamination.py` — 4-point CosMx physical plaque audit (`402` plaques), 69-section proximal cell counts, and Tier-1 spillover decontamination.
- `run_everything_requested.py` — 27-donor demographics, 5-fold donor `GridSearchCV`, 16-regime Monte Carlo simulation, and 4-method changepoint benchmark.
- `run_four_critical_checks.py` — `adata.X` matrix normalization verification, donor-CV feature ablation, 27-donor OOF Table S4, and CosMx CP1K total-transcript normalization.
- `run_subclass_and_within_section_check.py` — Tier-2 24-subclass expression & spillover audit (`1.89M` cells) and strictly within-section multicellular niche verification.
- `run_donor_level_gradient_test.py` — 27-donor patient-level Mann-Whitney U tests (`Dementia`, `APOE4`) and Spearman correlation with `CPS`.
- `generate_all_16_figures.py` — Regenerates all 8 Main Figures (`Figure_1`–`Figure_8`) and 8 Supplementary Figures (`SuppFig_S1`–`SuppFig_S8`) at 300 DPI.
- `generate_graphical_abstract.py` — Generates `Graphical_Abstract.png` at 300 DPI.

### Precomputed Summary Tables (CSV)
- `donor_cv_ablation_study.csv` — Table 1 feature ablation metrics across all 8 model configurations.
- `seaad_27_donors_demographics.csv` — Supplementary Table S1 (27-donor demographics & neuropathology).
- `TableS2_Full_140_Gene_Spillover_and_SHAP_Audit.csv` — Full 140-gene Tier-1 spillover & SHAP audit.
- `TableS3_All_69_Cortical_Sections_Proximal_Audit.csv` — All 69 cortical sections proximal cell counts & knots.
- `TableS4_27_Donors_OOF_Performance.csv` — Per-donor out-of-fold ROC-AUC, PR-AUC, and LogLoss across all 27 donors.
- `xgboost_grid_search_donor_cv.csv` — 5-fold donor `GridSearchCV` hyperparameter tuning results.
- `simulation_study_results.csv` — 16-regime Monte Carlo power, bias, and FPR simulation table.
- `method_benchmark_comparison.csv` — 4-method changepoint benchmark (`Hinge`, `GAM`, `PELT`, `Binseg`).
- `early_warning_genes.csv` — Differential expression screening (`25–50 µm` vs. `>50 µm`).

---

## Quickstart & Full Reproduction

```bash
pip install -r requirements.txt

# 1. Download public datasets (SEA-AD MERFISH & CosMx GSE263791)
python download_data.py
python download_cosmx_plaque.py

# 2. Run full multi-platform audit & verification suite
python run_full_reanalysis.py
python run_advanced_upgrades.py
python run_critical_audit_and_decontamination.py
python run_everything_requested.py
python run_four_critical_checks.py
python run_subclass_and_within_section_check.py
python run_donor_level_gradient_test.py

# 3. Generate all 16 publication figures + Graphical Abstract (300 DPI)
python generate_all_16_figures.py
python generate_graphical_abstract.py
```

## License
Released under the MIT License (see `LICENSE`).
