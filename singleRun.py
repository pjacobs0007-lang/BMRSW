import argparse
import numpy as np
import os
from BMRSW import drawSingleBootstrapSample

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CLI for BMRSW Single Run on Raw Dataset (No Bootstrapping)")
    parser.add_argument('--lam', type=float, required=True, help='Lambda parameter')
    parser.add_argument('--input_dataset', type=str, required=True, help='Path to .npy dataset')
    parser.add_argument('--model', type=str, required=True, help='Model simulator name (e.g., gandk, normal)')
    
    parser.add_argument('--p_bounds', type=float, nargs='+', required=True, help='Bounds as min max min max...')
    parser.add_argument('--x0', type=float, nargs='+', required=True, help='Initial position for CMA-ES')
    
    # CMA-ES params
    parser.add_argument('--popsize', type=int, default=16)
    parser.add_argument('--max_gen', type=int, default=100)
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--sigma0', type=float, default=None)
    
    # Other parameters
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sga_iter', type=int, default=10000)
    parser.add_argument('--sga_batch', type=int, default=1)
    parser.add_argument('--burn_in_pct', type=float, default=60.0)
    
    # Output directory
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = np.load(args.input_dataset)
    
    # Parse p_bounds
    if len(args.p_bounds) % 2 != 0:
        raise ValueError("p_bounds must be a list of even length (min1 max1 min2 max2...)")
    p_bounds_parsed = [(args.p_bounds[i], args.p_bounds[i+1]) for i in range(0, len(args.p_bounds), 2)]
    
    print(f"Starting Single Run (no bootstrapping) with lambda={args.lam}...")
    
    best_x, X_np = drawSingleBootstrapSample(
        dataset=dataset, lam=args.lam, popsize=args.popsize, max_gen=args.max_gen, 
        tol=args.tol, n_jobs=args.n_jobs, model=args.model, p_bounds=p_bounds_parsed, 
        x0=args.x0, seed=args.seed, sga_iter=args.sga_iter, sga_batch=args.sga_batch, 
        burn_in_pct=args.burn_in_pct, sigma0=args.sigma0, output_dir=args.output_dir,
        bootstrap_idx=0, resample=False
    )
    
    print(f"Single Run Completed! Estimated theta: {best_x}")
