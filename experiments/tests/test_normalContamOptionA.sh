#!/bin/bash
cd "$(dirname "$0")/../.."

CONTAM_OPTION="A"
N=1000

echo "Running Data Creation for Contam Option ${CONTAM_OPTION}"
python3 datasets/normalDataCreation.py \
    --theta_true 0 1 \
    --N ${N} \
    --R 1 \
    --contamOption ${CONTAM_OPTION} \
    --eps 0.05 \
    --nu 22 \
    --z 10

DATASET_PATH="datasets/normal/${N}/contamOption=${CONTAM_OPTION}/dataset_0.npy"

LAM_SELECT_DIR="results/normal/contamOption${CONTAM_OPTION}/lamSelect"
echo "Running BMRSW in lam_select mode"
python3 BMRSW.py \
    --mode lam_select \
    --input_dataset ${DATASET_PATH} \
    --M 2 \
    --lams 2.5 3.5 \
    --model normal \
    --p_bounds -10 10 0.1 20 \
    --x0 -5 0.15 \
    --sga_iter 50 \
    --sga_batch 1 \
    --burn_in_pct 60 \
    --popsize 4 \
    --max_gen 2 \
    --sigma0 1 \
    --tol 0.000001 \
    --output_dir ${LAM_SELECT_DIR}

SINGLERUN_DIR="results/normal/bmrsw/contamOption${CONTAM_OPTION}/singleRun"
echo "Running singleRun.py (raw dataset, no bootstrapping)"
python3 singleRun.py \
    --lam 2.5 \
    --input_dataset ${DATASET_PATH} \
    --model normal \
    --p_bounds -10 10 0.1 20 \
    --x0 -5 0.15 \
    --sga_iter 50 \
    --sga_batch 1 \
    --burn_in_pct 60 \
    --popsize 4 \
    --max_gen 2 \
    --sigma0 1 \
    --tol 0.000001 \
    --output_dir ${SINGLERUN_DIR}

PERFORMUQ_DIR="results/normal/bmrsw/contamOption${CONTAM_OPTION}/performUQ"
echo "Running BMRSW in performUQ mode"
python3 BMRSW.py \
    --mode performUQ \
    --lam 2.5 \
    --input_dataset ${DATASET_PATH} \
    --M 2 \
    --model normal \
    --p_bounds -10 10 0.1 20 \
    --x0 -5 0.15 \
    --sga_iter 50 \
    --sga_batch 1 \
    --burn_in_pct 60 \
    --popsize 4 \
    --max_gen 2 \
    --sigma0 1 \
    --tol 0.000001 \
    --output_dir ${PERFORMUQ_DIR}

WASSER_DIR="results/normal/wasser/contamOption${CONTAM_OPTION}"
mkdir -p ${WASSER_DIR}
echo "Running bernton_minWasser"
python3 bernton_minWasser.py \
    --dataset ${DATASET_PATH} \
    --model normal \
    --loMu -10 --hiMu 10 --initMu -5 \
    --loSig 0.1 --hiSig 20 --initSig 0.15 \
    --K 10 \
    --M_sim 100 \
    --num_bootstraps 2 \
    --output_dir ${WASSER_DIR}

IMG_DIR="results/normal/images"
mkdir -p ${IMG_DIR}
echo "Running viz_creator_normal.py"
python3 viz_creator_normal.py \
    --model normal \
    --N ${N} \
    --contamOption ${CONTAM_OPTION} \
    --eps 0.05 \
    --nu 22 \
    --z 10 \
    --lamsForPlot 2.5 \
    --output ${IMG_DIR}/test_contamOption${CONTAM_OPTION}.pdf
