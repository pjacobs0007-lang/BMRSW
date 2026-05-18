import os
import json
import argparse
import numpy as np
import time
from NPL import npl
from models import g_and_k_model, normal_model

def main():
    parser = argparse.ArgumentParser(description="Run Production NPL MMD Study on a single dataset")
    parser.add_argument('--input_dataset', type=str, required=True, help='Path to .npy dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--model', type=str, required=True, choices=['gandk', 'normal'], help='Model name (gandk or normal)')
    
    parser.add_argument('--x0', type=float, nargs='+', default=None, help='Initial position for Adam optimizer')
    
    # Study Params
    parser.add_argument('--bandwidth', type=float, default=-1.0, help='RBF Kernel bandwidth (-1 for median heuristic)')
    parser.add_argument('--B', type=int, default=100, help='Number of bootstrap samples')
    parser.add_argument('--m', type=int, default=500, help='Number of simulated samples per parameter')
    parser.add_argument('--batch_size', type=int, default=500, help='Mini-batch size for Adam optimization')
    parser.add_argument('--Nstep', type=int, default=1000, help='Number of Adam steps')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading dataset from {args.input_dataset}")
    X = np.load(args.input_dataset)
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
        
    N = X.shape[0]
    
    if args.model == 'gandk':
        model = g_and_k_model(m=args.m, d=1)
        p = 4
        x0 = args.x0 if args.x0 is not None else [5., 5., 5., 5.]
    elif args.model == 'normal':
        model = normal_model(m=args.m, d=1)
        p = 2
        x0 = args.x0 if args.x0 is not None else [-5.0, 0.15]
    else:
        raise ValueError(f"Unknown model: {args.model}")

    print(f"Starting NPL Production Study | model={args.model}, N={N}, bandwidth={args.bandwidth}, B={args.B}, m={args.m}")

    total_start_time = time.time()
    
    # Initialize NPL
    npl_inference = npl(X=X, 
                        B=args.B, 
                        m=args.m, 
                        p=p, 
                        l=args.bandwidth, 
                        model=model, 
                        model_name=args.model,
                        batch_size=args.batch_size,
                        x0=x0,
                        Nstep=args.Nstep)

    # Run Bayesian Bootstrap
    npl_inference.draw_samples()
    
    samples = np.array(npl_inference.sample) # Shape (B, p)
    
    for b in range(args.B):
        result = {
            "bootstrap_idx": b,
            "model": args.model,
            "theta_est": samples[b].tolist(),
            "final_f": 0.0,
            "params": vars(args)
        }
        
        output_filename = os.path.join(args.output_dir, f"bootstrap_idx_{b}.json")
        with open(output_filename, 'w') as f:
            json.dump(result, f, indent=4)
            
    total_elapsed = time.time() - total_start_time
    print(f"Study Finished! Total time: {total_elapsed/60:.2f} minutes.")
    print(f"Bootstrap means: {np.mean(samples, axis=0)}")
    print(f"Results saved to {args.output_dir}/")

if __name__ == '__main__':
    main()
