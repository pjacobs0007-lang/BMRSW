#!/bin/bash
cd "$(dirname "$0")/../.." 

echo "============================================================"
echo " Stage 1: Data Creation for Test G-and-K Coverage Experiment"
echo "============================================================"

# Just N=1000, R=2
python3 datasets/gandkDataCreation.py \
    --theta_true 3 1 2 0.5 \
    --R 2 \
    --eps 0.05 \
    --z_outlier 50 \
    --rho 0.05 \
    --N 1000 \
    --no_log_k \
    --seed 42

echo "============================================================"
echo " Stage 2: BMRSW Uncertainty Quantification (performUQ)"
echo "============================================================"

for N in 1000; do
    # Just 1 lambda
    for LAM_VAL in 0.5; do
        for i in {0..1}; do
        python3 BMRSW.py \
            --mode performUQ \
            --lam ${LAM_VAL} \
            --input_dataset datasets/gandk/N=${N}/dataset_${i}.npy \
            --M 2 \
            --model gandk \
            --p_bounds -10 10 0.1 10 0.03 40.0 0.05 3.0 \
            --x0 5.0 0.15 0.05 0.05 \
            --sga_iter 50 \
            --sga_batch 1 \
            --burn_in_pct 60 \
            --popsize 4 \
            --max_gen 2 \
            --tol 0.000001 \
            --sigma0 1 \
            --no_log_k \
            --output_dir results/gandk/bmrsw/N${N}/performUQ/lam_${LAM_VAL}/dataset_${i}
        done
    done
done

echo "============================================================"
echo " Stage 3: NPL MMD Uncertainty Quantification"
echo "============================================================"

for N in 1000; do
    for BW_VAL in -1; do
        for i in {0..1}; do
            python3 npl_mmd/run_NPL_MMD.py \
                --input_dataset datasets/gandk/N=${N}/dataset_${i}.npy \
                --output_dir results/gandk/npl_mmd/N${N}/bandwidth_${BW_VAL}/dataset_${i} \
                --model gandk \
                --bandwidth ${BW_VAL} \
                --B 2 \
                --m 20 \
                --batch_size 200 \
                --Nstep 50
        done
    done
done

echo "============================================================"
echo " Stage 4: Diagnostic Visualization Generation"
echo "============================================================"

IMG_DIR="results/gandk/images"
mkdir -p ${IMG_DIR}

python3 viz_creator_gandk.py \
    --theta_true 3 1 2 0.5 \
    --eps 0.05 \
    --rho 0.05 \
    --z_outlier 50 \
    --base_dir results/gandk \
    --output_path ${IMG_DIR}/test_gandk_diagnostic_panel_part3.pdf \
    --output_table ${IMG_DIR}/test_gandk_comparison_table.tex
