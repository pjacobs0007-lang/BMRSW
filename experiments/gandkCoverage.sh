#!/bin/bash
cd "$(dirname "$0")/.."

echo "============================================================"
echo " Stage 1: Data Creation for G-and-K Coverage Experiment"
echo "============================================================"


echo "Generating N=1000 datasets..."
python datasets/gandkDataCreation.py \
    --theta_true 3 1 2 0.5 \
    --R 20 \
    --eps 0.05 \
    --z_outlier 50 \
    --rho 0.05 \
    --N 1000 \
    --no_log_k \
    --seed 42

echo "Generating N=5000 datasets..."
python datasets/gandkDataCreation.py \
    --theta_true 3 1 2 0.5 \
    --R 20 \
    --eps 0.05 \
    --z_outlier 50 \
    --rho 0.05 \
    --N 5000 \
    --no_log_k \
    --seed 42

echo "Data creation complete!"

echo "============================================================"
echo " Stage 2: BMRSW Uncertainty Quantification (performUQ)"
echo "============================================================"

for N in 1000 5000; do
    for LAM_VAL in 0.5 1.5 2.5 3.5; do
        echo "Processing datasets for N=${N} with Lambda=${LAM_VAL}..."
        
        # Process all 20 datasets
        for i in {0..19}; do
        echo "  -> Running BMRSW on dataset_${i}.npy"
        
        python BMRSW.py \
            --mode performUQ \
            --lam ${LAM_VAL} \
            --input_dataset datasets/gandk/N=${N}/dataset_${i}.npy \
            --M 100 \
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
            --output_dir results/gandk/bmrsw/N${N}/performUQ/lam_${LAM_VAL}/dataset_${i}
            
        done
    done
done

echo "============================================================"
echo " Stage 3: NPL MMD Uncertainty Quantification"
echo "============================================================"

for N in 1000 5000; do
    for BW_VAL in -1 0.15 0.3 0.5 1; do
        echo "Processing datasets for N=${N} with Bandwidth=${BW_VAL}..."
        
        for i in {0..19}; do
            echo "  -> Running NPL MMD on dataset_${i}.npy"
            
            python npl_mmd/run_NPL_MMD.py \
                --input_dataset datasets/gandk/N=${N}/dataset_${i}.npy \
                --output_dir results/gandk/npl_mmd/N${N}/bandwidth_${BW_VAL}/dataset_${i} \
                --model gandk \
                --bandwidth ${BW_VAL} \
                --B 100 \
                --m 20 \
                --batch_size 200 \
                --Nstep 1000
                
        done
    done
done

echo "============================================================"
echo " Stage 4: Diagnostic Visualization Generation"
echo "============================================================"

IMG_DIR="results/gandk/images"
mkdir -p ${IMG_DIR}

echo "Running viz_creator_gandk.py..."
python viz_creator_gandk.py \
    --theta_true 3 1 2 0.5 \
    --eps 0.05 \
    --rho 0.05 \
    --z_outlier 50 \
    --base_dir results/gandk \
    --output_path ${IMG_DIR}/gandk_diagnostic_panel_part3.pdf \
    --output_table ${IMG_DIR}/gandk_comparison_table.tex

echo "Coverage Experiment Complete!"
