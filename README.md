# LEMURS Correlation & Profile Analysis

A single, git-ready script to compute and visualize calorimeter shower correlations and profiles across layers, angular (ϕ) bins, and radial (r) bins. It supports both NPZ and HDF5 inputs and produces clear, multi-page PDF reports.

**Data**: For the experiment, download the dataset from **Zenodo**: [https://zenodo.org/records/17045562](https://zenodo.org/records/17045562).


---

## Table of Contents

- [Features](#features)
- [Input Formats](#input-formats)
  - [.npz](#npz)
  - [.h5 / .hdf5](#h5--hdf5)
- [Installation](#installation)
- [Quickstart](#quickstart)
  - [Stratified correlation matrices](#1-stratified-correlation-matrices-multi-page-pdf)
  - [Stratified mean profiles](#2-stratified-mean-profiles-optionally-per-event-normalized)
  - [ϕ/r-bin shift analyses](#3-φr-bin-shift-analyses-layer-like)
  - [Layer-wise shift analysis](#4-layer-wise-shift-analysis-original-idea)
  - [Global (all-data) correlations](#5-global-all-data-correlations-for-layersϕr)
- [Command Reference](#command-reference)
  - [`corr`](#corr)
  - [`means`](#means)
  - [`phi-r`](#phi-r)
  - [`layer-shifts`](#layer-shifts)
  - [`global-corr`](#global-corr)
- [What is `--min-samples`?](#what-is---min-samples)
- [Outputs](#outputs)
- [Tips](#tips)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

- **Stratified correlation matrices**  
  - Layers (45×45), ϕ-bins (16×16), r-bins (9×9)  
  - Stratified by incident **energy (E)**, **polar angle (θ)**, and **azimuth (ϕ)**
- **Stratified mean ± SEM profiles**  
  - Layers / ϕ / r, with optional per-event normalization
- **Shift analyses**  
  - **Layer-wise** (original idea): mean/std and adjacent-pair correlation shifts across θ/ϕ bins  
  - **ϕ-bin / r-bin** shift analyses (parity with layer-wise)
- **Global (all-data) correlations**  
  - Single non-stratified matrices for layers / ϕ / r

All outputs are saved as PDFs.

---

## Input Formats

### `.npz`

Arrays (any of the accepted aliases are fine):

- `X` or `showers` or `shower`: **(N, 45, 16, 9)**  
- `E_inc` or `incident_energy` or `E`: **(N,)**  
- `theta` or `incident_theta`: **(N,)** in radians  
- `phi` or `incident_phi`: **(N,)** in radians  
- Optional: `U_true` **(N,45)** (per-layer energy). If present, the `layer-shifts` command will use it.

### `.h5 / .hdf5`

Datasets:

- `showers`: **(N, R, Phi, Z)** → the script **transposes** to **(N, Z, Phi, R) = (N, 45, 16, 9)**  
- `incident_energy`: **(N,)**  
- `incident_theta`: **(N,)** in radians  
- `incident_phi`: **(N,)** in radians

> **Axis convention used by the script:** **(N, Z=45, ϕ=16, r=9)**

---

## Installation

```bash
# Python 3.9+ recommended
pip install numpy matplotlib seaborn h5py
# torch is optional; only needed if you pass torch tensors in your own code
```

---

## Quickstart



### 1) Stratified correlation matrices (multi-page PDF)

```bash
python lemurs_analysis.py corr --data path/to/file.h5   --n-energy-bins 10 --n-theta-bins 10 --n-phi-bins 10   --min-samples 50 --save stratified_correlations.pdf
```

### 2) Stratified mean profiles (optionally per-event normalized)

```bash
python lemurs_analysis.py means --data path/to/file.h5   --n-energy-bins 10 --n-theta-bins 10 --n-phi-bins 10   --min-samples 50 --normalize-per-event   --save mean_profiles_stratified.pdf
```

### 3) ϕ/r-bin shift analyses (layer-like)

```bash
python lemurs_analysis.py phi-r --data path/to/file.h5   --n-theta-bins 10 --n-phi-bins 4 --min-samples 50   --save-phi angular_bin_shifts_analysis.pdf   --save-r   radial_bin_shifts_analysis.pdf
```

### 4) Layer-wise shift analysis (original idea)

```bash
python lemurs_analysis.py layer-shifts --data path/to/file.h5   --n-theta-bins 10 --n-phi-bins 4 --min-samples 50   --save layer_shifts_analysis.pdf
```

### 5) Global (all-data) correlations for layers/ϕ/r

```bash
python lemurs_analysis.py global-corr --data path/to/file.h5   --save global_correlations.pdf
```

---

## Command Reference

All commands accept `--data` with `.npz` or `.h5/.hdf5` and `--dataset` with `cc2` for the CaloChallenge 2022 dataset and `lemurs` for the LEMURS dataset.


### `corr`

Stratified correlation matrices (Layers/ϕ/r) → PDF

- `--n-energy-bins INT` (default 3)  
- `--n-theta-bins INT` (default 3)  
- `--n-phi-bins INT` (default 3)  
- `--min-samples INT` (default 50)  
- `--annotate` (optional, show numbers inside heatmaps)  
- `--ncols INT` (default 3, layout columns)  
- `--save PATH` (PDF path)

### `means`

Stratified mean profiles (Layers/ϕ/r) → PDF

- `--n-energy-bins INT` (default 3)  
- `--n-theta-bins INT` (default 3)  
- `--n-phi-bins INT` (default 3)  
- `--min-samples INT` (default 50)  
- `--percentile-bins` (energy bins by percentiles instead of linear)  
- `--normalize-per-event` (scale each event by its total energy)  
- `--no-ci` (hide SEM shading)  
- `--ncols INT`  
- `--save PATH`

### `phi-r`

ϕ/r-bin shift analyses (mean/std + adjacent-corr) → PDFs

- `--n-theta-bins INT` (default 10)  
- `--n-phi-bins INT` (default 4)  
- `--min-samples INT` (default 50)  
- `--normalize-per-event` (optional)  
- `--save-phi PATH`, `--save-r PATH`

### `layer-shifts`

Layer-wise mean/std and adjacent-corr shifts vs θ/ϕ → PDF

- Uses `U_true (N,45)` if present in `.npz`, otherwise sums `X` over (ϕ, r)  
- `--n-theta-bins INT` (default 10)  
- `--n-phi-bins INT` (default 4)  
- `--min-samples INT` (default 50)  
- `--no-logy` (disable log y-axis on mean energy plots)  
- `--save PATH`

### `global-corr`

All-data (non-stratified) correlations for Layers/ϕ/r → PDF

- `--annotate` (optional)  
- `--save PATH`

---

## What is `--min-samples`?

When we stratify by **E**, **θ**, or **ϕ**, some bins may be sparse.  
`--min-samples` is a **quality threshold**: bins with fewer events than this value are **skipped** to avoid noisy or misleading panels.

- Default: **50**  
- Rule of thumb: ensure `N / (#bins)` is comfortably ≥ `min_samples`.  
  For large datasets, values like **50–200** usually work well.

If you need every bin plotted regardless of statistics, lower `--min-samples` (but expect noisier estimates).

---

## Outputs

All commands produce one or more **PDFs** with consistent styling:

- Heatmaps use symmetric color scale (Pearson r in **[-1, 1]**, `coolwarm`) and a **shared colorbar** per page.
- Profile plots optionally use **log-scale** on the y-axis (controllable in commands).
- Titles include the bin range and **N** for traceability.

---

## Tips

- **Axis sanity**: HDF5 `showers (N,R,Phi,Z)` is transposed to `(N,Z,Phi,R)` internally—no action needed.  
- **Per-event normalization** lets you compare **shape** changes across strata, independent of total energy.  
- **Layout**: use `--ncols` to keep grids readable when you have many bins.

---

## Troubleshooting

- **“Expected (N,45,16,9)”**: Check your input shapes after transpose (HDF5) or the array names (NPZ).  
- **Empty pages / few panels**: Your `--min-samples` may be too high or your bins too fine. Reduce bins or lower the threshold.  
- **All zeros / NaNs**: This can happen if a column has zero variance; the script replaces NaN entries with 0 and sets the diagonal to 1 for correlations.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
