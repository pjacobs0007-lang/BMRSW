#!/bin/bash
cd "$(dirname "$0")"

DATASET="datasets/gandk/N=5000/dataset_0.npy"
OUT_DIR="results/gandk/timing_test_npl"

echo "================================================"
echo " Timing NPL-MMD (1 Bootstrap, N=5000)"
echo "   m: 20"
echo "   batch_size: 200"
echo "   Nstep: 1000"
echo "================================================"

time python3 npl_mmd/run_NPL_MMD.py \
    --input_dataset ${DATASET} \
    --output_dir ${OUT_DIR} \
    --model gandk \
    --bandwidth -1 \
    --B 1 \
    --m 20 \
    --batch_size 200 \
    --Nstep 1000

echo "================================================"
echo " Timing test complete. Results in ${OUT_DIR}"
echo "================================================"
