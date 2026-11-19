# A Spillover-Decontaminated Spatial Probability Framework and Multi-Platform Benchmark for Plaque-Associated Myeloid Activation in Alzheimer's Disease

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Official repository for the manuscript:
> **A Spillover-Decontaminated Spatial Probability Framework and Multi-Platform Benchmark Reveal Intrinsic Myeloid Drivers and Pitfalls of Transcriptomic Plaque Proxies in Alzheimer's Disease**  
> Benjamin Goodluck Otuya (Department of Human Physiology, Ahmadu Bello University, Zaria, Nigeria)

## Overview
This repository provides an end-to-end, reproducible computational pipeline for modeling microglial spatial activation probability, auditing transcriptomic plaque-proxy anchors, filtering neighbor-proximity transcript spillover across 1.89 million human cortical cells, and validating peri-plaque decay boundaries against ground-truth anti-Aβ immunofluorescence plaques:

1. **Human Discovery & Audit Cohort (`SEA-AD MTG MERFISH`):** `1,887,729` total cortical cells (`195,620` `Microglia-PVM` cells across `27` human donors and `69` cortical sections; `140`-gene panel).
2. **Direct Plaque Validation Cohort (`NCBI GEO GSE263791` NanoString CosMx):** `43,117` cells (`11,434` microglia and `402` physical Aβ plaque object centroids co-registered with `MOAB-2` anti-Aβ immunofluorescence across `4` coronal brain hemisphere sections from 18-month *App*$^{NL-G-F}$ and C57BL/6J mice; Mallach et al., *Cell Reports* 2024).
3. **Monte Carlo Simulation & Method Benchmarking Suite:** Evaluates statistical power, false positive rate (FPR), and Mean Absolute Error (MAE) of Two-Piece Segmented Hinge Regression against GAM Cubic B-Splines, 2-Segment PELT Changepoint Detection, and Binary Step-Threshold (`Binseg`) scanning.

## Repository Contents
- `02_Spatial_Probability_Model_Complete_Reanalysis.ipynb` — Master executed Jupyter notebook containing Parts I–IV (baseline replication, negative controls, CosMx direct plaque validation, spatial autocorrelation control, donor-level `GroupKFold` cross-validation, spillover decontamination, and simulation study).
- `download_data.py` & `download_cosmx_plaque.py` — Automated scripts to download `SEAAD_MTG_MERFISH.2024-12-11.h5ad` from the public Allen Institute AWS S3 bucket (`s3://sea-ad-spatial-transcriptomics/`) and `GSM8199188`/`GSM8199189` from NCBI GEO (`GSE263791`).
- `run_full_reanalysis.py` — Full pipeline script for baseline GMM thresholding, XGBoost Spatial Probability Classifier, 2D SHAP, and Reviewer Analyses 1–7.
- `run_advanced_upgrades.py` — Executes `Supertype` myeloid lineage deconvolution, `APOE4` and clinical dementia stratification, and 1.89M-cell multicellular niche profiling.
- `run_critical_audit_and_decontamination.py` — Executes the 4-point CosMx plaque-centroid audit ($N=402$ plaques, $d=0$ excluded, plaque-cluster bootstrap CI), SEA-AD within-section proximal cell counts, and neighbor-spillover decontamination.
- `run_everything_requested.py` — Executes the 27-donor demographics extraction, 5-fold donor-level `GridSearchCV`, the Monte Carlo simulation study, and 4-method changepoint benchmarking.
- `requirements.txt` — Pinned Python package dependencies.
- `LICENSE` — MIT License.

## Quickstart & Reproduction
```bash
pip install -r requirements.txt
python download_data.py
python download_cosmx_plaque.py
python run_full_reanalysis.py
python run_advanced_upgrades.py
python run_critical_audit_and_decontamination.py
python run_everything_requested.py
```
