#!/bin/bash
#SBATCH -n 1
#SBATCH --job-name=LEMURS_corr6
#SBATCH -t 1:00:00
#SBATCH --mem=80G
#SBATCH -p bii-gpu
#SBATCH --gres=gpu:1
#SBATCH -A bii_nssac

set -euo pipefail

# --- Paths & params (edit these) ---
#DATA="/project/bi_dsc_community/calorimeter/LEMURS/Par04SiW_gamma/LEMURS_Par04SiW_gamma_100kEvents_1GeV1TeV_GPSflat_part1.h5"
DATA="/project/bi_dsc_community/calorimeter/calorimeter_evaluation_data/dataset_2/dataset_2_electron_Geant4.h5"
OUTDIR="./LEMURS_analysis_results/Calochallenge_ds2"
DATASET='cc2'
NB_ENERGY=10
NB_THETA=0
NB_PHI=0
MIN_SAMPLES=50
NCOLS=3            # columns per page in the PDF grid
ANNOTATE=""        # set to "--annotate" if you want numeric values drawn in each cell

# Timestamp for unique filenames
STAMP="$(date +%Y%m%d_%H%M%S)"

# --- Env setup ---
module load miniforge
module load texlive
source activate torch_gpu_renew

mkdir -p "${OUTDIR}"

# Small helper to run one configuration
run_corr () {
  local tag="$1"; shift
  local extra_flags=("$@")
  local save_path="${OUTDIR}/stratified_correlations_${tag}_${STAMP}.pdf"

  echo "[run] ${tag} -> ${save_path}"
  python lemurs_analysis.py \
    --dataset "${DATASET}" \
    corr \
        --data "${DATA}" \
        --n-energy-bins "${NB_ENERGY}" \
        --n-theta-bins "${NB_THETA}" \
        --n-phi-bins "${NB_PHI}" \
        --min-samples "${MIN_SAMPLES}" \
        --ncols "${NCOLS}" \
        ${ANNOTATE:+--annotate} \
        --save "${save_path}" \
        "${extra_flags[@]}"
}

# -------------------------
# Six patterns to generate:
# 1) RAW + Pearson
# 2) RAW + Spearman
# 3) NORM(per-event) + Pearson
# 4) NORM(per-event) + Spearman
# 5) LOG1P + Pearson
# 6) LOG1P + Spearman
# (Intentionally NOT doing LOG1P+NORM combos, to keep it at six as requested.)
# -------------------------

# 1) RAW + Pearson (no extra flags: Pearson is default)
run_corr "raw_Pearson"

# 2) RAW + Spearman
run_corr "raw_Spearman"             --spearman

# 3) NORM + Pearson
run_corr "norm_Pearson"             --normalize-per-event

# 4) NORM + Spearman
run_corr "norm_Spearman"            --normalize-per-event --spearman

# 5) LOG1P + Pearson
run_corr "log1p_Pearson"            --log1p

# 6) LOG1P + Spearman
run_corr "log1p_Spearman"           --log1p --spearman

echo "[done] All six PDFs written to ${OUTDIR}"
