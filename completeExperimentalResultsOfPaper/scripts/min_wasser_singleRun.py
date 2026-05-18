"""
min_wasser_singleRun.py
-----------------------
Single-run worker for Minimum Wasserstein (W2) inference.

For each objective evaluation at theta, the W2(P_n, P_theta) is estimated as:
    W2_est(theta) = (1/K) * sum_{i=1}^{K} W2(P_n, P_{theta,i,M})
where P_{theta,i,M} is the empirical measure of M i.i.d. draws from P_theta.

Nelder-Mead (scipy.optimize.minimize) is run on W2_est to produce a point estimate.

Supports:
  --model gandk   : G-and-K distribution (4 parameters: A, B, g, k)
  --model normal  : Normal (2 parameters: mu, sigma)

Task mapping (production mode, same as gandkSingleRun.py):
  dataset_index = task_id // 100
  seed_for_boot = task_id

Results are uploaded to:
  gs://<bucket>/<model>/results/N<N>/minW2/contamOption=<X>/task_<id>.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.optimize import minimize
import ot

try:
    from google.cloud import storage
except ImportError:
    storage = None

from differentiable_simulators import GandKSimulator, NormalMeanSimulator

torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# GCS Helpers (same as gandkSingleRun.py)
# ---------------------------------------------------------------------------
def upload_to_gcs(local_path, bucket_name, gcs_path):
    if storage is None:
        return False
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        return True
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        return False


def download_from_gcs(bucket_name, gcs_path, local_path):
    if storage is None:
        return False
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.download_to_filename(local_path)
        return True
    except Exception as e:
        print(f"Error downloading from GCS: {e}")
        return False


# ---------------------------------------------------------------------------
# Simulator factory
# ---------------------------------------------------------------------------
def make_simulator(model: str, model_kwargs: dict):
    if model == 'gandk':
        return GandKSimulator(d=1, **model_kwargs)
    elif model == 'normal':
        return NormalMeanSimulator(d=1, **model_kwargs)
    else:
        raise ValueError(f"Unknown model: '{model}'")


# ---------------------------------------------------------------------------
# W2 objective
# ---------------------------------------------------------------------------
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

    Returns 1e6 for out-of-bounds or numerically invalid parameters.
    """
    # Bounds penalty (Nelder-Mead ignores bounds natively in older scipy)
    for val, (lo, hi) in zip(theta, p_bounds):
        if val < lo or val > hi:
            return 1e6

    theta_t = (torch.tensor(theta, dtype=torch.float32)
               .unsqueeze(0)           # (1, d_param)
               .expand(M, -1))         # (M, d_param)

    w2_vals = []
    try:
        with torch.no_grad():
            for k in range(K):
                # Use pre-sampled noise for determinism
                noise = fixed_noise[k]
                samples = sim(theta_t, noise).numpy().flatten().astype(np.float64)
                w2_vals.append(ot.wasserstein_1d(X_np, samples, p=2))
    except Exception as e:
        print(f"Simulator error at theta={theta}: {e}")
        return 1e6

    val = float(np.mean(w2_vals))
    if np.isnan(val) or np.isinf(val):
        return 1e6
    return val


# ---------------------------------------------------------------------------
# Debug: 2-D grid search and contour plot (Normal model only)
# ---------------------------------------------------------------------------
def run_debug_grid(args, sim, X_np: np.ndarray) -> None:
    """
    Evaluates W2_est(mu, sigma) on a regular grid and saves a filled-contour
    PDF so the optimisation landscape can be inspected visually.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    mu_grid  = np.linspace(args.grid_mu_lo,  args.grid_mu_hi,  args.grid_n_mu)
    sig_grid = np.linspace(args.grid_sig_lo, args.grid_sig_hi, args.grid_n_sig)
    Z = np.full((args.grid_n_sig, args.grid_n_mu), np.nan)

    total = args.grid_n_mu * args.grid_n_sig
    done  = 0
    print(f"[debug_grid] Evaluating {total} grid points "
          f"(K={args.K}, M={args.M})...", flush=True)

    p_bounds = [(args.loMu, args.hiMu), (args.loSig, args.hiSig)]
    for j, sig in enumerate(sig_grid):
        for i, mu in enumerate(mu_grid):
            Z[j, i] = w2_objective(np.array([mu, sig]), X_np,
                                   sim, args.K, args.M, p_bounds)
            done += 1
            if done % max(1, total // 20) == 0:
                print(f"  {done}/{total} ({100*done/total:.0f}%)", flush=True)

    # --- Plot ---
    MU, SIG = np.meshgrid(mu_grid, sig_grid)

    fig, ax = plt.subplots(figsize=(8, 6))
    levels  = np.linspace(np.nanpercentile(Z, 2), np.nanpercentile(Z, 98), 40)
    cf = ax.contourf(MU, SIG, Z, levels=levels, cmap='viridis_r')
    cs = ax.contour(MU,  SIG, Z, levels=levels[::4], colors='white',
                    linewidths=0.5, alpha=0.5)
    fig.colorbar(cf, ax=ax, label=r'$\hat{W}_2(P_n, P_\theta)$')

    # Grid minimum
    min_idx  = np.unravel_index(np.nanargmin(Z), Z.shape)
    mu_min   = mu_grid[min_idx[1]]
    sig_min  = sig_grid[min_idx[0]]
    ax.plot(mu_min, sig_min, 'r*', markersize=14, label=f'Grid min ({mu_min:.3f}, {sig_min:.3f})')

    # True theta
    if len(args.theta_true) >= 2:
        ax.plot(args.theta_true[0], args.theta_true[1], 'w^',
                markersize=10, label=rf'$\theta^*$ ({args.theta_true[0]}, {args.theta_true[1]})')

    # Empirical mean of data (vertical line on mu axis)
    emp_mean = float(np.mean(X_np))
    ax.axvline(emp_mean, color='orange', linestyle='--', linewidth=1.5,
               label=rf'Empirical mean ({emp_mean:.3f})')

    ax.set_xlabel(r'$\mu$', fontsize=13)
    ax.set_ylabel(r'$\sigma$', fontsize=13)
    ax.set_title(rf'$\hat{{W}}_1$ landscape — {args.model}, N={args.N_data}'
                 + (f', contamOption={args.NormalContamOption}'
                    if args.NormalContamOption else ''),
                 fontsize=13)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    out = args.debug_output or f"w2_landscape_{args.model}.pdf"
    plt.savefig(out, bbox_inches='tight', dpi=150)
    print(f"[debug_grid] Saved contour plot to: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Minimum Wasserstein (W2) Single-Run Worker"
    )

    # 1. Input / Output
    parser.add_argument('--ground_truth_data', type=str, required=True,
                        help='Path to .npy dataset (local or gs://)')
    parser.add_argument('--bucket', type=str, default=None,
                        help='GCS bucket name for result upload')
    parser.add_argument('--results_prefix', type=str, default=None,
                        help='GCS path prefix for results (overrides default minW2 layout)')
    parser.add_argument('--task_id', type=int, default=-1)
    parser.add_argument('--N_data', type=int, default=1000,
                        help='Sample size (used in output GCS path)')

    # 2. Model
    parser.add_argument('--model', type=str, default='gandk',
                        choices=['gandk', 'normal'],
                        help='Model to fit')
    parser.add_argument('--NormalContamOption', type=str, default=None,
                        choices=['A', 'B', 'C'],
                        help='Contamination option; sets GCS sub-directory')

    # 3. Bounds & Init — G-and-K
    parser.add_argument('--loA',   type=float, default=0.0)
    parser.add_argument('--hiA',   type=float, default=5.0)
    parser.add_argument('--initA', type=float, default=None)
    parser.add_argument('--loB',   type=float, default=0.0)
    parser.add_argument('--hiB',   type=float, default=2.0)
    parser.add_argument('--initB', type=float, default=None)
    parser.add_argument('--log',   type=float, default=0.0)
    parser.add_argument('--hig',   type=float, default=5.0)
    parser.add_argument('--initg', type=float, default=None)
    parser.add_argument('--lok',   type=float, default=0.0)
    parser.add_argument('--hik',   type=float, default=2.0)
    parser.add_argument('--initk', type=float, default=None)

    # 4. Bounds & Init — Normal
    parser.add_argument('--loMu',   type=float, default=-5.0)
    parser.add_argument('--hiMu',   type=float, default=5.0)
    parser.add_argument('--initMu', type=float, default=None)
    parser.add_argument('--loSig',   type=float, default=0.01)
    parser.add_argument('--hiSig',   type=float, default=5.0)
    parser.add_argument('--initSig', type=float, default=None)

    # 5. Ground-truth parameters
    parser.add_argument('--theta_true', type=float, nargs='+', default=None,
                        help='True parameters: 4 for gandk, 2 for normal')
    parser.add_argument('--no_log_k', action='store_true',
                        help='Use raw k (not log_k) for G-and-K')

    # 6. W2 estimation hyperparameters
    parser.add_argument('--K', type=int, default=10,
                        help='Number of Monte Carlo replicates per W2 evaluation')
    parser.add_argument('--M', type=int, default=1000,
                        help='Simulator samples per replicate')

    # 7. Nelder-Mead options
    # (uses scipy defaults: xatol=fatol=1e-4, maxiter=200*n_params)

    # 8. Debug / Grid Search Mode
    parser.add_argument('--debug_grid', action='store_true',
                        help='Run a 2-D grid search over (mu, sigma) and plot '
                             'W2 contours. Skips Nelder-Mead and bootstrapping. '
                             'Normal model only.')
    parser.add_argument('--grid_mu_lo',  type=float, default=-1.0)
    parser.add_argument('--grid_mu_hi',  type=float, default=7.0)
    parser.add_argument('--grid_n_mu',   type=int,   default=30)
    parser.add_argument('--grid_sig_lo', type=float, default=0.5)
    parser.add_argument('--grid_sig_hi', type=float, default=3.0)
    parser.add_argument('--grid_n_sig',  type=int,   default=30)
    parser.add_argument('--debug_output', type=str, default=None,
                        help='Output path for the contour PDF (debug_grid mode).')

    args = parser.parse_args()

    # --- Validate / default theta_true ---
    expected_len = 4 if args.model == 'gandk' else 2
    if args.theta_true is None:
        args.theta_true = ([3.0, 1.0, 2.0, 0.5] if args.model == 'gandk'
                           else [0.0, 1.0])
    elif len(args.theta_true) != expected_len:
        parser.error(
            f"--theta_true requires {expected_len} values for "
            f"model='{args.model}', got {len(args.theta_true)}"
        )

    # --- Build model_kwargs ---
    if args.model == 'gandk':
        model_kwargs = {'use_log_k': not args.no_log_k}
    else:
        model_kwargs = {'sigma_fixed': None, 'parameterization': 'linear'}

    # --- Debug grid: load dataset_0 raw, evaluate W2 landscape, plot, exit ---
    if args.debug_grid:
        if args.model != 'normal':
            print("Error: --debug_grid is only supported for --model normal.")
            sys.exit(1)
        sim = make_simulator(args.model, model_kwargs)
        torch.manual_seed(0)

        # Resolve dataset_0 path (strip trailing .npy / _ then append _0.npy)
        base = args.ground_truth_data
        if base.endswith('.npy'):
            base = base[:-4]
        if base.endswith('_'):
            base = base[:-1]
        data_path = f"{base}_0.npy"
        local_path = "data_temp_debug.npy"

        if data_path.startswith("gs://"):
            bkt  = data_path.split("/")[2]
            blob = "/".join(data_path.split("/")[3:])
            if not download_from_gcs(bkt, blob, local_path):
                print(f"Failed to download {data_path}")
                sys.exit(1)
        else:
            local_path = data_path

        X_np = np.load(local_path).astype(np.float64).flatten()
        print(f"[debug_grid] Loaded dataset_0: N={len(X_np)}, "
              f"mean={X_np.mean():.4f}, std={X_np.std():.4f}")
        run_debug_grid(args, sim, X_np)
        sys.exit(0)

    # --- Task identification ---
    task_id = args.task_id
    if task_id == -1 and 'BATCH_TASK_INDEX' in os.environ:
        task_id = int(os.environ['BATCH_TASK_INDEX'])
    if task_id == -1:
        print("Error: No task_id or BATCH_TASK_INDEX found.")
        sys.exit(1)

    # --- Production task mapping (same as gandkSingleRun.py production mode) ---
    dataset_index = task_id // 100
    seed_for_boot = task_id
    print(f"Task {task_id}: Min-W2 Production Mode "
          f"(Dataset {dataset_index}, Seed {seed_for_boot})", flush=True)

    # --- Data loading ---
    base_path = args.ground_truth_data
    if base_path.endswith('.npy'):
        base_path = base_path[:-4]
    if base_path.endswith('_'):
        base_path = base_path[:-1]

    full_gcs_path   = f"{base_path}_{dataset_index}.npy"
    local_data_path = f"data_temp_w2_{task_id}.npy"

    if full_gcs_path.startswith("gs://"):
        bucket_candidate = full_gcs_path.split("/")[2]
        gcs_blob_path    = "/".join(full_gcs_path.split("/")[3:])
        if not download_from_gcs(bucket_candidate, gcs_blob_path, local_data_path):
            print(f"Failed to download dataset from {full_gcs_path}")
            sys.exit(1)
    else:
        local_data_path = full_gcs_path

    full_data = np.load(local_data_path)
    N = len(full_data)

    # --- Bootstrap ---
    np.random.seed(seed_for_boot)
    indices = np.random.choice(N, size=N, replace=True)
    X_np    = full_data[indices].astype(np.float64).flatten()
    print(f"Task {task_id}: Bootstrapped {N} samples from {args.ground_truth_data}")

    # --- Bounds and initial point ---
    if args.model == 'gandk':
        p_bounds = [
            (args.loA, args.hiA),
            (args.loB, args.hiB),
            (args.log, args.hig),
            (args.lok, args.hik),
        ]
        x0 = [
            args.initA if args.initA is not None else (args.loA + args.hiA) / 2.0,
            args.initB if args.initB is not None else (args.loB + args.hiB) / 2.0,
            args.initg if args.initg is not None else (args.log + args.hig) / 2.0,
            args.initk if args.initk is not None else (args.lok + args.hik) / 2.0,
        ]
    else:  # normal
        p_bounds = [
            (args.loMu,  args.hiMu),
            (args.loSig, args.hiSig),
        ]
        x0 = [
            args.initMu  if args.initMu  is not None else (args.loMu  + args.hiMu)  / 2.0,
            args.initSig if args.initSig is not None else (args.loSig + args.hiSig) / 2.0,
        ]

    # --- Build simulator ---
    torch.manual_seed(task_id)
    sim = make_simulator(args.model, model_kwargs)

    # --- Pre-sample noise for deterministic optimization ---
    # This prevents Nelder-Mead from getting stuck on stochastic "dips"
    noise_dim = 1 # both gandk and normal are 1D simulators in this script
    torch.manual_seed(task_id)
    fixed_noise = torch.randn(args.K, args.M, noise_dim)

    # --- Optimize ---
    print(f"Starting Nelder-Mead: x0={x0}, K={args.K}, M={args.M} (Deterministic fixed noise)", flush=True)
    start_time = time.time()

    n_evals = [0]
    def objective_with_logging(theta):
        n_evals[0] += 1
        val = w2_objective(theta, X_np, sim, args.K, args.M, p_bounds, fixed_noise)
        if n_evals[0] % 20 == 0:
            print(f"  Task {task_id} | eval {n_evals[0]:4d}: "
                  f"theta={np.round(theta, 4)}, W2={val:.6f}", flush=True)
        return val

    result = minimize(
        objective_with_logging,
        x0,
        method='Nelder-Mead',
        bounds=p_bounds,          # requires scipy >= 1.7
        options={'maxfev': 2000, 'maxiter': 1000}
    )

    total_time = time.time() - start_time
    best_x = result.x

    print(f"Task {task_id} complete. "
          f"theta_est={np.round(best_x, 5)}, "
          f"W2={result.fun:.6f}, "
          f"evals={n_evals[0]}, "
          f"time={total_time:.1f}s", flush=True)

    # --- GCS output path ---
    if args.results_prefix is not None:
        results_prefix = args.results_prefix
    elif args.NormalContamOption is not None:
        results_prefix = (f"{args.model}/results/N{args.N_data}/minW2"
                          f"/contamOption={args.NormalContamOption}")
    else:
        results_prefix = f"{args.model}/results/N{args.N_data}/minW2"

    # --- Save result ---
    result_dict = {
        "task_id":      int(task_id),
        "model":        str(args.model),
        "dataset_id":   int(dataset_index),
        "K":            args.K,
        "M":            args.M,
        "theta_est":    [float(v) for v in best_x],
        "final_w2":     float(result.fun),
        "n_evals":      int(n_evals[0]),
        "n_iter":       int(result.nit) if hasattr(result, 'nit') else None,
        "success":      bool(result.success),
        "message":      str(result.message),
        "total_time_s": float(total_time),
        "params":       vars(args),
    }

    output_filename = f"task_{task_id}.json"
    with open(output_filename, 'w') as f:
        json.dump(result_dict, f, indent=4)

    if args.bucket:
        gcs_tgt = f"{results_prefix}/{output_filename}"
        print(f"Uploading result to gs://{args.bucket}/{gcs_tgt}...")
        upload_to_gcs(output_filename, args.bucket, gcs_tgt)

    # --- Cleanup temp data ---
    try:
        if (os.path.exists(local_data_path)
                and args.ground_truth_data.startswith("gs://")):
            os.remove(local_data_path)
    except Exception as e:
        print(f"Cleanup warning: {e}")


if __name__ == "__main__":
    main()
