import argparse
import numpy as np
import os
import torch
import sys

# Ensure the parent directory (code/) is in the path so we can import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import sample_gandk_groundTruth

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Generate Ground Truth Datasets for G-and-K Study")
    parser.add_argument('--theta_true', type=float, nargs=4, default=[3.0, 1.0, 2.0, 0.5])
    parser.add_argument('--eps', type=float, default=0.04)
    parser.add_argument('--z_outlier', type=float, default=50.0)
    parser.add_argument('--rho', type=float, default=0.5)
    parser.add_argument('--N', type=int, default=10000)
    parser.add_argument('--no_log_k', action='store_true', help='Use absolute scale for k instead of log(k)')
    parser.add_argument('--R', type=int, default=20, help='Number of unique datasets to create')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default=None, help='Local output directory')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join("datasets", "gandk", f"N={args.N}")
        
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    print(f"Generating {args.R} datasets with theta_true={args.theta_true}...")
    
    # Ensure reproducibility for base datasets
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    for i in range(args.R):
        # We vary the seed slightly for each unique dataset
        curr_seed = args.seed + i
        np.random.seed(curr_seed)
        torch.manual_seed(curr_seed)
        
        X_np = sample_gandk_groundTruth(
            args.theta_true[0], args.theta_true[1], args.theta_true[2], args.theta_true[3],
            args.eps, args.z_outlier, args.rho, args.N, 
            use_log_k=(not args.no_log_k)
        ).ravel().astype(np.float64)
        
        fname = os.path.join(args.output_dir, f"dataset_{i}.npy")
        np.save(fname, X_np)
        print(f"Saved {fname}")

if __name__ == "__main__":
    main()
