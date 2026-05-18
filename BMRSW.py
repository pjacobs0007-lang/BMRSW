import numpy as np
import torch
import time
import cma
import argparse
import os
import json
from joblib import Parallel, delayed
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None

try:
    import ot
except ImportError:
    ot = None

from simulators import GandKSimulator, NormalMeanSimulator
from sga import original_sga_solve

def _eval_point(theta_nd, X_np, pre_fixed_noises,
                lam, burn_in_pct=0.0, model='gandk', model_kwargs=None):
    """Evaluates a single parameter point using SGA."""
    theta_nd = np.array(theta_nd, dtype=np.float32)
    try:
        # Note: eval_noise_np was removed from original_sga_solve in sga.py
        val, _ = original_sga_solve(
            theta_nd, X_np, pre_fixed_noises,
            lam, 0.0, model=model, model_kwargs=model_kwargs,
            burn_in_pct=burn_in_pct
        )

        if np.isnan(val) or np.isinf(val):
            return 1e6
        return float(val)
    except Exception as e:
        return 1e6

def drawSingleBootstrapSample(dataset, lam, popsize, max_gen, tol, n_jobs,
                              model, p_bounds, x0, seed,
                              sga_iter=10000, sga_batch=1, burn_in_pct=60.0,
                              sigma0=None, model_kwargs=None,
                              output_dir=None, bootstrap_idx=None, resample=True,
                              compute_w2=False):
    """
    Replicates exactly the singleRun logic in gandkSingleRun.py for a single bootstrap sample.
    
    Args:
        dataset (np.ndarray): The full dataset to bootstrap from.
        lam (float): The lambda parameter for the dual objective.
        popsize (int): Population size for CMA-ES.
        max_gen (int): Maximum number of generations for CMA-ES.
        tol (float): Tolerance (tolfun/tolx) for CMA-ES.
        n_jobs (int): Number of parallel jobs for evaluation.
        model (str): Model name ('gandk' or 'normal').
        p_bounds (list of tuples): List of (lower, upper) bounds for each parameter.
        x0 (list): Initial position for CMA-ES.
        seed (int): Random seed for reproducibility.
        sga_iter (int): Number of SGA iterations.
        sga_batch (int): SGA batch size (usually 1).
        burn_in_pct (float): Percentage of SGA trajectory to burn in.
        sigma0 (float): Initial step size for CMA-ES.
        model_kwargs (dict): Additional keyword arguments for the simulator.
        
    Returns:
        np.ndarray: The best parameters found by CMA-ES.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    N = len(dataset)
    # 1. Bootstrap Resampling
    if resample:
        indices = np.random.choice(N, size=N, replace=True)
        X_np = dataset[indices]
        print(f"Bootstrapped {N} samples.")
    else:
        X_np = dataset
        print(f"Using full dataset of {N} samples without bootstrapping.")
    
    # 2. Pre-generate noise for SGA
    # Note: slr was removed from the simplified simulators/sga, but we check for dimension consistency
    noise_dim = 2 if model == 'slr' else 1
    pre_noise_sga = np.random.randn(sga_iter * sga_batch, noise_dim).astype(np.float32)
    
    # 3. CMA-ES Setup
    if sigma0 is None:
        sigma0 = min([(b[1] - b[0]) / 4.0 for b in p_bounds])
        
    print(f"Starting CMA-ES with x0={x0}, sigma0={sigma0}")
    
    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'popsize': popsize,
        'bounds': [[b[0] for b in p_bounds], [b[1] for b in p_bounds]],
        'seed': seed,
        'tolfun': tol,
        'tolx': tol,
    })
    
    gen = 0
    while not es.stop() and gen < max_gen:
        gen += 1
        solutions = es.ask()
        nj = n_jobs if n_jobs > 0 else 4
        
        with Parallel(n_jobs=nj) as parallel:
            fitness_list = parallel(delayed(_eval_point)(
                x, X_np, pre_noise_sga,
                lam, burn_in_pct,
                model=model,
                model_kwargs=model_kwargs,
            ) for x in solutions)
        
        es.tell(solutions, fitness_list)
        best_f = es.result.fbest
        print(f"Gen {gen:02d}: f_best={best_f:.6f}")
        
    best_x = es.result.xbest
    best_f = es.result.fbest
    print(f"Optimization completed. Best point: {best_x}")
    
    w2_val = None
    if compute_w2:
        w2_val = _compute_w2_fit(best_x, X_np, lam, model, model_kwargs)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        result = {
            "bootstrap_idx": int(bootstrap_idx) if bootstrap_idx is not None else None,
            "resample": resample,
            "model": str(model),
            "lambda": float(lam),
            "theta_est": [float(x) for x in best_x],
            "final_f": float(best_f),
            "seed": int(seed) if seed is not None else None,
            "p_bounds": p_bounds,
            "x0": x0
        }
        if w2_val is not None:
            result["w2"] = float(w2_val)
        
        if resample and bootstrap_idx is not None:
            out_file = os.path.join(output_dir, f"bootstrap_lam_{lam}_idx_{bootstrap_idx}.json")
        else:
            out_file = os.path.join(output_dir, f"singleRun_lam_{lam}.json")
            
        with open(out_file, 'w') as f:
            json.dump(result, f, indent=4)
            
    return best_x, X_np, w2_val

def _compute_w2_fit(best_x, X_np, lam, model, model_kwargs=None):
    """Computes the Wasserstein-2 distance between the fitted model and the reweighted sample."""
    if ot is None:
        return None
        
    # 1. Get final dual variables g
    # We use a large number of iterations for the diagnostic noise if needed, 
    # but here we'll just reuse the SGA logic.
    noise_dim = 2 if model == 'slr' else 1
    diag_noise = np.random.randn(10000, noise_dim).astype(np.float32)
    
    _, g_final = original_sga_solve(
        best_x, X_np, diag_noise,
        lam, 0.0, model=model, model_kwargs=model_kwargs
    )
    
    # 2. Compute reweighted distribution weights sigma_w
    v = -lam * g_final
    v -= v.max()
    sigma_w = np.exp(v)
    sigma_w /= sigma_w.sum()
    
    # 3. Sample from fitted model
    batch_size_w2 = 1000
    theta_t = torch.tensor(best_x, dtype=torch.float32).unsqueeze(0).expand(batch_size_w2, -1)
    
    if model == 'gandk':
        sim = GandKSimulator(d=1, **(model_kwargs or {}))
        noise_w2 = torch.randn(batch_size_w2, 1)
    elif model == 'normal':
        sim = NormalMeanSimulator(d=1, sigma_fixed=None, **(model_kwargs or {}))
        noise_w2 = torch.randn(batch_size_w2, 1)
    else:
        # Fallback/Other models
        return None
        
    with torch.no_grad():
        Z_fit = sim(theta_t, noise_w2).numpy().astype(np.float64)
    
    # 4. Compute W2^2
    M_dist = (Z_fit - X_np.reshape(1, -1)) ** 2
    a_weights = np.ones(batch_size_w2) / batch_size_w2
    b_weights = sigma_w.astype(np.float64)
    b_weights /= b_weights.sum()
    
    try:
        w2_sq = ot.emd2(a_weights, b_weights, M_dist, numItermax=1000000)
        return np.sqrt(w2_sq)
    except Exception as e:
        print(f"Error computing W2: {e}")
        return None

def performUQ(dataset, lam, popsize, max_gen, tol, n_jobs,
              model, p_bounds, x0, seed, M, p,
              sga_iter=10000, sga_batch=1, burn_in_pct=60.0,
              sigma0=None, model_kwargs=None, output_dir=None):
    """
    Performs Uncertainty Quantization by running M bootstrap samples.
    
    Returns:
        tuple: (ci, estimates)
            - ci (np.ndarray): Lower and upper bounds of the (1-p)*100% marginal central credible intervals. Shape is (2, n_params).
            - estimates (np.ndarray): Full matrix of bootstrap estimates. Shape is (M, n_params).
    """
    estimates = []
    
    print(f"Starting UQ analysis with M={M} bootstrap samples...")
    for i in range(M):
        print(f"\n--- Bootstrap Sample {i+1}/{M} ---")
        # Ensure each bootstrap has a unique but reproducible seed
        boot_seed = seed + i if seed is not None else None
        
        est, _, _ = drawSingleBootstrapSample(
            dataset=dataset, lam=lam, popsize=popsize, max_gen=max_gen, 
            tol=tol, n_jobs=n_jobs, model=model, p_bounds=p_bounds, 
            x0=x0, seed=boot_seed, sga_iter=sga_iter, sga_batch=sga_batch, 
            burn_in_pct=burn_in_pct, sigma0=sigma0, model_kwargs=model_kwargs,
            output_dir=output_dir, bootstrap_idx=i, compute_w2=False
        )
        estimates.append(est)
        
    estimates = np.array(estimates)
    
    # Calculate marginal central credible intervals
    alpha = p
    lower_q = (alpha / 2.0) * 100
    upper_q = (1.0 - alpha / 2.0) * 100
    
    lower_bounds = np.percentile(estimates, lower_q, axis=0)
    upper_bounds = np.percentile(estimates, upper_q, axis=0)
    
    ci = np.vstack([lower_bounds, upper_bounds])
    
    print("\nUQ Analysis Completed.")
    print(f"{(1-p)*100:.1f}% Marginal Central Credible Intervals:")
    for i in range(estimates.shape[1]):
        print(f" Parameter {i}: [{ci[0, i]:.4f}, {ci[1, i]:.4f}]")
        
    return ci, estimates

def performLamSelect(dataset, popsize, max_gen, tol, n_jobs,
                     model, p_bounds, x0, seed, M, lams=None,
                     sga_iter=10000, sga_batch=1, burn_in_pct=60.0,
                     sigma0=None, model_kwargs=None, output_dir=None):
    """
    Performs Lambda Selection by evaluating a range of lambdas, each with M bootstrap samples.
    Returns a dictionary where each lambda is a key and the value is an array of W2 fit values.
    """
    if lams is None:
        lams = np.logspace(-2, 2, 15)
    
    results = {}
    overall_boot_idx = 0
    
    print(f"Starting Lambda Selection with {len(lams)} values, M={M} samples each...")
    
    for lam in lams:
        print(f"\n========== Lambda: {lam:.4f} ==========")
        results[lam] = []
        
        for m in range(M):
            print(f"\n--- Lambda {lam:.4f} | Bootstrap {m+1}/{M} ---")
            # Unique seed for each (lambda, bootstrap) pair if desired, 
            # or just unique per bootstrap. Here we use an overall counter.
            boot_seed = seed + overall_boot_idx if seed is not None else None
            overall_boot_idx += 1
            
            # 1. Draw bootstrap and optimize (and compute W2 inside)
            best_x, X_boot, w2 = drawSingleBootstrapSample(
                dataset=dataset, lam=lam, popsize=popsize, max_gen=max_gen,
                tol=tol, n_jobs=n_jobs, model=model, p_bounds=p_bounds,
                x0=x0, seed=boot_seed, sga_iter=sga_iter, sga_batch=sga_batch,
                burn_in_pct=burn_in_pct, sigma0=sigma0, model_kwargs=model_kwargs,
                output_dir=output_dir, bootstrap_idx=overall_boot_idx, compute_w2=True
            )
            
            if w2 is not None:
                results[lam].append(w2)
                print(f"W2 Fit: {w2:.6f}")
        
        results[lam] = np.array(results[lam])
        
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CLI for BMRSW")
    parser.add_argument('--mode', type=str, required=True, choices=['lam_select', 'performUQ'])
    parser.add_argument('--lam', type=float, default=None, help='Lambda parameter (required for performUQ)')
    parser.add_argument('--lams', type=float, nargs='+', default=None, help='List of lambdas (used for lam_select)')
    parser.add_argument('--input_dataset', type=str, required=True, help='Path to .npy dataset')
    parser.add_argument('--M', type=int, required=True, help='Number of bootstrap samples')
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
    parser.add_argument('--no_log_k', action='store_true', help='Use absolute scale for k (gandk only)')
    
    # UQ specific
    parser.add_argument('--p', type=float, default=0.05, help='Significance level for UQ (e.g., 0.05 for 95% CI)')

    # Output directory
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = np.load(args.input_dataset)
    
    # Parse p_bounds
    if len(args.p_bounds) % 2 != 0:
        raise ValueError("p_bounds must be a list of even length (min1 max1 min2 max2...)")
    p_bounds_parsed = [(args.p_bounds[i], args.p_bounds[i+1]) for i in range(0, len(args.p_bounds), 2)]
    
    if args.mode == 'performUQ':
        if args.lam is None:
            raise ValueError("--lam is required for performUQ mode")
            
        model_kwargs = {}
        if args.model == 'gandk' and args.no_log_k:
            model_kwargs['use_log_k'] = False
        
        ci, ests = performUQ(
            dataset=dataset, lam=args.lam, popsize=args.popsize, max_gen=args.max_gen, 
            tol=args.tol, n_jobs=args.n_jobs, model=args.model, p_bounds=p_bounds_parsed, 
            x0=args.x0, seed=args.seed, M=args.M, p=args.p,
            sga_iter=args.sga_iter, sga_batch=args.sga_batch, burn_in_pct=args.burn_in_pct,
            sigma0=args.sigma0, model_kwargs=model_kwargs, output_dir=args.output_dir
        )
        print("Final Credible Intervals:\n", ci)
    
    elif args.mode == 'lam_select':
        if args.lams:
            print(f"lam_select mode called with lambdas: {args.lams}")
        else:
            print("lam_select mode called. Currently using default lambdas.")
            
        model_kwargs = {}
        if args.model == 'gandk' and args.no_log_k:
            model_kwargs['use_log_k'] = False
            
        res = performLamSelect(
            dataset=dataset, popsize=args.popsize, max_gen=args.max_gen,
            tol=args.tol, n_jobs=args.n_jobs, model=args.model, p_bounds=p_bounds_parsed,
            x0=args.x0, seed=args.seed, M=args.M, lams=args.lams,
            sga_iter=args.sga_iter, sga_batch=args.sga_batch, burn_in_pct=args.burn_in_pct,
            sigma0=args.sigma0, model_kwargs=model_kwargs, output_dir=args.output_dir
        )
        print("Lambda Selection Results:\n", res)

