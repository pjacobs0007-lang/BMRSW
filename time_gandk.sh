#!/bin/bash
cd "$(dirname "$0")"

echo "================================================"
echo " Stage 1: Generating N=5000 Dataset"
echo "================================================"
python3 datasets/gandkDataCreation.py \
    --theta_true 3 1 2 0.5 \
    --R 1 \
    --eps 0.05 \
    --z_outlier 50 \
    --rho 0.05 \
    --N 5000 \
    --no_log_k \
    --seed 42

DATASET="datasets/gandk/N=5000/dataset_0.npy"
OUT_DIR="results/gandk/timing_test"

echo "================================================"
echo " Stage 2: Timing BMRSW (1 Bootstrap, N=5000)"
echo "   popsize: 16"
echo "   max_gen (rounds): 50"
echo "   sga_iter: 20000"
echo "================================================"

time python3 BMRSW.py \
    --mode performUQ \
    --lam 1.5 \
    --input_dataset ${DATASET} \
    --M 1 \
    --model gandk \
    --p_bounds -10 10 0.1 10 0.03 40.0 0.05 3.0 \
    --x0 5.0 0.15 0.05 0.05 \
    --sga_iter 20000 \
    --sga_batch 1 \
    --burn_in_pct 60 \
    --popsize 16 \
    --max_gen 50 \
    --tol 0.000001 \
    --sigma0 1 \
    --no_log_k \
    --output_dir ${OUT_DIR}

echo "================================================"
echo " Timing test complete. Results in ${OUT_DIR}"
echo "================================================"
