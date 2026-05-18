import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.optimize import minimize
import ot

from simulators import GandKSimulator, NormalMeanSimulator

torch.set_num_threads(1)

def make_simulator(model: str, model_kwargs: dict):
    if model == 'gandk':
        return GandKSimulator(d=1, **(model_kwargs or {}))
    elif model == 'normal':
        return NormalMeanSimulator(d=1, **(model_kwargs or {}))
    else:
        raise ValueError(f"Unknown model: '{model}'")

def w2_objective(theta: np.ndarray,
                 X_np: np.ndarray,
                 sim,
                 K: int,
                 M: int,
                 p_bounds: list,
                 fixed_noise: torch.Tensor) -> float:
    """
    Estimates W2(P_n, P_theta) by averaging K independent W2(P_n, P_{theta,i,M}).
    Uses fixed_noise to make the objective deterministic.
    """
    for val, (lo, hi) in zip(theta, p_bounds):
        if val < lo or val > hi:
            return 1e6

    theta_t = (torch.tensor(theta, dtype=torch.float32)
               .unsqueeze(0)
               .expand(M, -1))

    w2_vals = []
    try:
        with torch.no_grad():
            for k in range(K):
                noise = fixed_noise[k]
                samples = sim(theta_t, noise).numpy().flatten().astype(np.float64)
                w2_vals.append(ot.wasserstein_1d(X_np, samples, p=2))
    except Exception as e:
        return 1e6

    val = float(np.mean(w2_vals))
    if np.isnan(val) or np.isinf(val):
        return 1e6
    return val

def drawSingleBootstrapSample(dataset, K, M_sim, model, p_bounds, x0, seed, model_kwargs=None,
                              output_dir=None, bootstrap_idx=None):
    """
    Performs the Minimum Wasserstein inference for a single bootstrap sample.
    Returns the estimated parameters and the bootstrapped dataset.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    N = len(dataset)
    indices = np.random.choice(N, size=N, replace=True)
    X_np = dataset[indices]
    
    sim = make_simulator(model, model_kwargs)
    
    noise_dim = 1
    fixed_noise = torch.randn(K, M_sim, noise_dim)
    
    def objective_func(theta):
        return w2_objective(theta, X_np, sim, K, M_sim, p_bounds, fixed_noise)
        
    result = minimize(
        objective_func,
        x0,
        method='Nelder-Mead',
        bounds=p_bounds,
        options={'maxfev': 2000, 'maxiter': 1000}
    )
    
    if output_dir is not None and bootstrap_idx is not None:
        os.makedirs(output_dir, exist_ok=True)
        out_dict = {
            "bootstrap_idx": int(bootstrap_idx),
            "model": str(model),
            "theta_est": [float(x) for x in result.x],
            "final_f": float(result.fun),
            "seed": int(seed) if seed is not None else None,
            "p_bounds": p_bounds,
            "x0": x0
        }
        out_file = os.path.join(output_dir, f"bootstrap_idx_{bootstrap_idx}.json")
        with open(out_file, 'w') as f:
            json.dump(out_dict, f, indent=4)
            
    return result.x, X_np

def performUQ(dataset, K, M_sim, model, p_bounds, x0, seed, num_bootstraps, p_ci, model_kwargs=None, output_dir=None):
    """
    Performs Uncertainty Quantization by running num_bootstraps bootstrap samples.
    
    Returns:
        tuple: (ci, estimates)
            - ci (np.ndarray): Lower and upper bounds of the (1-p_ci)*100% marginal central credible intervals. Shape is (2, n_params).
            - estimates (np.ndarray): Full matrix of bootstrap estimates. Shape is (num_bootstraps, n_params).
    """
    estimates = []
    
    print(f"Starting Min-Wasserstein UQ analysis with {num_bootstraps} bootstrap samples...")
    for i in range(num_bootstraps):
        boot_seed = seed + i if seed is not None else None
        print(f"\n--- Bootstrap Sample {i+1}/{num_bootstraps} ---")
        
        est, _ = drawSingleBootstrapSample(
            dataset=dataset, K=K, M_sim=M_sim, model=model, 
            p_bounds=p_bounds, x0=x0, seed=boot_seed, 
            model_kwargs=model_kwargs,
            output_dir=output_dir, bootstrap_idx=i
        )
        estimates.append(est)
        print(f"  Estimated theta: {np.round(est, 4)}")
        
    estimates = np.array(estimates)
    
    # Calculate marginal central credible intervals
    alpha = p_ci
    lower_q = (alpha / 2.0) * 100
    upper_q = (1.0 - alpha / 2.0) * 100
    
    lower_bounds = np.percentile(estimates, lower_q, axis=0)
    upper_bounds = np.percentile(estimates, upper_q, axis=0)
    
    ci = np.vstack([lower_bounds, upper_bounds])
    
    print("\nUQ Analysis Completed.")
    print(f"{(1-p_ci)*100:.1f}% Marginal Central Credible Intervals:")
    for i in range(estimates.shape[1]):
        print(f" Parameter {i}: [{ci[0, i]:.4f}, {ci[1, i]:.4f}]")
        
    return ci, estimates

def main():
    parser = argparse.ArgumentParser(description="Minimum Wasserstein (W2) UQ CLI")
    
    parser.add_argument('--dataset', type=str, required=True, help='Path to the .npy dataset')
    parser.add_argument('--model', type=str, default='gandk', choices=['gandk', 'normal'], help='Model to fit')
    
    # Bounds & Init — G-and-K
    parser.add_argument('--loA', type=float, default=0.0)
    parser.add_argument('--hiA', type=float, default=5.0)
    parser.add_argument('--initA', type=float, default=None)
    parser.add_argument('--loB', type=float, default=0.0)
    parser.add_argument('--hiB', type=float, default=2.0)
    parser.add_argument('--initB', type=float, default=None)
    parser.add_argument('--log', type=float, default=0.0)
    parser.add_argument('--hig', type=float, default=5.0)
    parser.add_argument('--initg', type=float, default=None)
    parser.add_argument('--lok', type=float, default=0.0)
    parser.add_argument('--hik', type=float, default=2.0)
    parser.add_argument('--initk', type=float, default=None)

    # Bounds & Init — Normal
    parser.add_argument('--loMu', type=float, default=-5.0)
    parser.add_argument('--hiMu', type=float, default=5.0)
    parser.add_argument('--initMu', type=float, default=None)
    parser.add_argument('--loSig', type=float, default=0.01)
    parser.add_argument('--hiSig', type=float, default=5.0)
    parser.add_argument('--initSig', type=float, default=None)
    
    # W2 Estimation
    parser.add_argument('--K', type=int, default=10, help='Number of Monte Carlo replicates per W2 evaluation')
    parser.add_argument('--M_sim', type=int, default=1000, help='Simulator samples per replicate (M in original)')
    
    # UQ parameters
    parser.add_argument('--num_bootstraps', type=int, default=100, help='Number of bootstrap samples to take')
    parser.add_argument('--p_ci', type=float, default=0.05, help='p value for (1-p)*100%% central credible intervals')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Other
    parser.add_argument('--no_log_k', action='store_true', help='Use raw k (not log_k) for G-and-K')
    parser.add_argument('--output_file', type=str, default='bernton_minWasser_results.json', help='Where to save results')
    parser.add_argument('--output_dir', type=str, required=True, help='Where to save per-bootstrap JSONs')
    
    args = parser.parse_args()
    
    # Data loading
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset {args.dataset} not found.")
        sys.exit(1)
        
    full_data = np.load(args.dataset).flatten().astype(np.float64)
    
    # Setup bounds and x0
    if args.model == 'gandk':
        p_bounds = [(args.loA, args.hiA), (args.loB, args.hiB), (args.log, args.hig), (args.lok, args.hik)]
        x0 = [
            args.initA if args.initA is not None else (args.loA + args.hiA) / 2.0,
            args.initB if args.initB is not None else (args.loB + args.hiB) / 2.0,
            args.initg if args.initg is not None else (args.log + args.hig) / 2.0,
            args.initk if args.initk is not None else (args.lok + args.hik) / 2.0,
        ]
        model_kwargs = {'use_log_k': not args.no_log_k}
    else:  # normal
        p_bounds = [(args.loMu, args.hiMu), (args.loSig, args.hiSig)]
        x0 = [
            args.initMu if args.initMu is not None else (args.loMu + args.hiMu) / 2.0,
            args.initSig if args.initSig is not None else (args.loSig + args.hiSig) / 2.0,
        ]
        model_kwargs = {'sigma_fixed': None, 'parameterization': 'linear'}
        
    start_time = time.time()
    
    ci, estimates = performUQ(
        dataset=full_data,
        K=args.K,
        M_sim=args.M_sim,
        model=args.model,
        p_bounds=p_bounds,
        x0=x0,
        seed=args.seed,
        num_bootstraps=args.num_bootstraps,
        p_ci=args.p_ci,
        model_kwargs=model_kwargs,
        output_dir=args.output_dir
    )
    
    total_time = time.time() - start_time
    
    results = {
        'model': args.model,
        'dataset': args.dataset,
        'K': args.K,
        'M_sim': args.M_sim,
        'num_bootstraps': args.num_bootstraps,
        'p_ci': args.p_ci,
        'credible_intervals_lower': ci[0].tolist(),
        'credible_intervals_upper': ci[1].tolist(),
        'bootstrap_estimates': estimates.tolist(),
        'total_time_s': total_time,
        'args': vars(args)
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nResults successfully saved to {args.output_file}")

if __name__ == "__main__":
    main()
