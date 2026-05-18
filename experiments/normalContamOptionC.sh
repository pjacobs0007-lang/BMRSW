#!/bin/bash
cd "$(dirname "$0")/.."

CONTAM_OPTION="C"
N=1000

echo "Running Data Creation for Contam Option ${CONTAM_OPTION}"
python datasets/normalDataCreation.py \
    --theta_true 0 1 \
    --N ${N} \
    --R 1 \
    --contamOption ${CONTAM_OPTION}

DATASET_PATH="datasets/normal/${N}/contamOption=${CONTAM_OPTION}/dataset_0.npy"

LAM_SELECT_DIR="results/normal/contamOption${CONTAM_OPTION}/lamSelect"
echo "Running BMRSW in lam_select mode"
python BMRSW.py \
    --mode lam_select \
    --input_dataset ${DATASET_PATH} \
    --M 6 \
    --model normal \
    --p_bounds -10 10 0.1 20 \
    --x0 -5 0.15 \
    --sga_iter 20000 \
    --sga_batch 1 \
    --burn_in_pct 60 \
    --popsize 16 \
    --max_gen 50 \
    --sigma0 1 \
    --tol 0.000001 \
    --output_dir ${LAM_SELECT_DIR}

SINGLERUN_DIR="results/normal/bmrsw/contamOption${CONTAM_OPTION}/singleRun"
echo "Running singleRun.py (raw dataset, no bootstrapping)"
python singleRun.py \
    --lam 2.5 \
    --input_dataset ${DATASET_PATH} \
    --model normal \
    --p_bounds -10 10 0.1 20 \
    --x0 -5 0.15 \
    --sga_iter 20000 \
    --sga_batch 1 \
    --burn_in_pct 60 \
    --popsize 16 \
    --max_gen 50 \
    --sigma0 1 \
    --tol 0.000001 \
    --output_dir ${SINGLERUN_DIR}

PERFORMUQ_DIR="results/normal/bmrsw/contamOption${CONTAM_OPTION}/performUQ"
echo "Running BMRSW in performUQ mode"
python BMRSW.py \
    --mode performUQ \
    --lam 2.5 \
    --input_dataset ${DATASET_PATH} \
    --M 100 \
    --model normal \
    --p_bounds -10 10 0.1 20 \
    --x0 -5 0.15 \
    --sga_iter 20000 \
    --sga_batch 1 \
    --burn_in_pct 60 \
    --popsize 16 \
    --max_gen 50 \
    --sigma0 1 \
    --tol 0.000001 \
    --output_dir ${PERFORMUQ_DIR}

WASSER_DIR="results/normal/wasser/contamOption${CONTAM_OPTION}"
mkdir -p ${WASSER_DIR}
echo "Running bernton_minWasser"
python bernton_minWasser.py \
    --dataset ${DATASET_PATH} \
    --model normal \
    --loMu -10 --hiMu 10 --initMu -5 \
    --loSig 0.1 --hiSig 20 --initSig 0.15 \
    --K 20 \
    --M_sim 10000 \
    --num_bootstraps 100 \
    --output_dir ${WASSER_DIR}

IMG_DIR="results/normal/images"
mkdir -p ${IMG_DIR}
echo "Running viz_creator_normal.py"
python viz_creator_normal.py \
    --model normal \
    --N ${N} \
    --contamOption ${CONTAM_OPTION} \
    --lamsForPlot 2.5 \
    --output ${IMG_DIR}/contamOption${CONTAM_OPTION}.pdf
