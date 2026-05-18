import argparse
import numpy as np
import torch
import time
import cma
import os
import sys
import json
from joblib import Parallel, delayed

try:
    from google.cloud import storage
except ImportError:
    storage = None

try:
    import jax
except ImportError:
    jax = None

torch.set_num_threads(1)  # Prevent torch from competing with joblib

try:
    import ot
except ImportError:
    ot = None

from original_sga_inner import original_sga_solve
from differentiable_simulators import GandKSimulator, NormalMeanSimulator

# ---------------------------------------------------------------------------
# GCS Helpers
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
# Evaluation Function
# ---------------------------------------------------------------------------
def _eval_point(theta_nd, X_np, pre_fixed_noises, eval_noise_np,
                eval_mode, lam, burn_in_pct=0.0, use_log_k=True,
                sim_instance=None, model='gandk', model_kwargs=None):
    """Evaluates a single parameter point using SGA."""
    if model_kwargs is None:
        # Backward-compatible fallback for gandk
        model_kwargs = {'use_log_k': use_log_k}
    theta_nd = np.array(theta_nd, dtype=np.float32)
    try:
        if eval_mode == 'sga':
            val, _ = original_sga_solve(
                theta_nd, X_np, pre_fixed_noises, eval_noise_np,
                lam, 0.0, model=model, model_kwargs=model_kwargs,
                burn_in_pct=burn_in_pct
            )
        else:
            raise ValueError(f"Unsupported eval_mode: {eval_mode}")

        if np.isnan(val) or np.isinf(val):
            return 1e6
        return float(val)
    except Exception as e:
        print(f"Eval error: {e}")
        return 1e6

# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Single Bootstrap Optimization Worker (gandk or normal)"
    )

    # 1. Input/Output
    parser.add_argument('--ground_truth_data', type=str, required=True,
                        help='Path to .npy dataset (local or GCS)')
    parser.add_argument('--bucket', type=str, default=None)
    parser.add_argument('--results_prefix', type=str, default="results_v2")
    parser.add_argument('--task_id', type=int, default=-1)

    # 2. Model
    parser.add_argument('--model', type=str, default='gandk',
                        choices=['gandk', 'normal'],
                        help='Model to fit (default: gandk)')

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

    # 4. Bounds & Init — Normal (mu, sigma)
    parser.add_argument('--loMu',   type=float, default=-5.0)
    parser.add_argument('--hiMu',   type=float, default=5.0)
    parser.add_argument('--initMu', type=float, default=None)
    parser.add_argument('--loSig',   type=float, default=0.01)
    parser.add_argument('--hiSig',   type=float, default=5.0)
    parser.add_argument('--initSig', type=float, default=None)

    # 5. Solver Params
    parser.add_argument('--theta_true', type=float, nargs='+', default=None,
                        help='True parameters: 4 values for gandk (a b g k), '
                             '2 for normal (mu sigma)')
    parser.add_argument('--eval_mode',   type=str,   default='sga')
    parser.add_argument('--sga_iter',    type=int,   default=10000)
    parser.add_argument('--sga_batch',   type=int,   default=1)
    parser.add_argument('--no_log_k',    action='store_true')
    parser.add_argument('--lam',         type=float, default=0.001)
    parser.add_argument('--burn_in_pct', type=float, default=0.6)

    # 6. Global Search (CMA-ES)
    parser.add_argument('--optMode',  type=str,   default='CMAES')
    parser.add_argument('--popsize',  type=int,   default=16)
    parser.add_argument('--max_gen',  type=int,   default=100)
    parser.add_argument('--sigma0',   type=float, default=None)
    parser.add_argument('--tolfun',   type=float, default=1e-6)
    parser.add_argument('--n_jobs',   type=int,   default=-1)

    # 7. Lambda Selection Mode
    parser.add_argument('--mode', type=str, default='production',
                        choices=['production', 'lam_select', 'getPointEstimate'])
    parser.add_argument('--numRepsLamSelectMode', type=int, default=10)
    parser.add_argument('--N_data', type=int, default=5000)
    parser.add_argument('--NormalContamOption', type=str, default=None,
                        choices=['A', 'B', 'C'],
                        help='Contamination option (A/B/C). '
                             'Required when --model normal and --mode lam_select '
                             '(sets the GCS sub-directory). '
                             'Also required for all models when --mode getPointEstimate.')

    # 8. Point Estimate Mode
    # (no extra args needed — the orchestrator passes --lam and --task_id directly)

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
    else:  # normal
        model_kwargs = {'sigma_fixed': None, 'parameterization': 'linear'}

    # --- Task Identification ---
    task_id = args.task_id
    if task_id == -1 and 'BATCH_TASK_INDEX' in os.environ:
        task_id = int(os.environ['BATCH_TASK_INDEX'])

    if task_id == -1:
        print("Error: No task_id or BATCH_TASK_INDEX found.")
        sys.exit(1)

    # --- Task & Data Mapping ---
    if args.mode == 'lam_select':
        dataset_index = 0
        boot_index = task_id // 15
        lam_index  = task_id % 15
        lams = np.logspace(-2, 2, 15)
        args.lam = float(lams[lam_index])
        seed_for_boot = boot_index
        print(f"Task {task_id}: Lambda Selection Mode "
              f"(Boot {boot_index}, Lam {args.lam:.4f})")
    elif args.mode == 'getPointEstimate':
        # The orchestrator submits one job per lambda, passing --lam directly.
        dataset_index = 0
        seed_for_boot = task_id  # used only for noise generation
        print(f"Task {task_id}: Point Estimate Mode "
              f"(Lam={args.lam:.6f}, no bootstrap)", flush=True)
    else:
        dataset_index = task_id // 100
        seed_for_boot = task_id
        print(f"Task {task_id}: Production Mode "
              f"(Dataset {dataset_index}, Seed {seed_for_boot})", flush=True)

    base_path = args.ground_truth_data
    if base_path.endswith(".npy"):
        base_path = base_path[:-4]
    if base_path.endswith("_"):
        base_path = base_path[:-1]

    full_gcs_path   = f"{base_path}_{dataset_index}.npy"
    local_data_path = f"data_temp_{task_id}.npy"

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

    # --- Bootstrap (skipped in getPointEstimate mode) ---
    if args.mode == 'getPointEstimate':
        # Use the raw dataset with no resampling so that weight index i
        # maps directly to full_data[i] for downstream visualization.
        X_np = full_data.copy()
        print(f"Task {task_id}: Using raw dataset (no bootstrap), N={N} samples",
              flush=True)
    else:
        np.random.seed(seed_for_boot)
        indices = np.random.choice(N, size=N, replace=True)
        X_np    = full_data[indices]
        print(f"Task {task_id}: Bootstrapped {N} samples from "
              f"{args.ground_truth_data}")

    # --- Optimization Setup ---
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

    # Pre-generate noises for the SGA inner loop
    torch.manual_seed(task_id)
    pre_noise_sga  = np.random.randn(args.sga_iter * args.sga_batch, 1).astype(np.float32)
    eval_noise_sga = np.random.randn(100, 1).astype(np.float32)

    sigma0 = args.sigma0 if args.sigma0 else min([(b[1] - b[0]) / 4.0 for b in p_bounds])

    print(f"Starting CMA-ES with x0={x0}, sigma0={sigma0}")
    start_time_exec = time.time()
    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'popsize': args.popsize,
        'bounds':  [[b[0] for b in p_bounds], [b[1] for b in p_bounds]],
        'seed':    task_id,
        'tolfun':  1e-12,
        'tolx':    1e-12,
    })

    gen, history = 0, []
    while not es.stop() and gen < args.max_gen:
        gen += 1
        solutions = es.ask()
        nj = args.n_jobs if args.n_jobs > 0 else 4
        with Parallel(n_jobs=nj) as parallel:
            print(
                f"Task {task_id} | Gen {gen:02d}: Evaluating {len(solutions)} points "
                f"using {nj} jobs (JAX={jax is not None})...", flush=True
            )
            fitness_list = parallel(delayed(_eval_point)(
                x, X_np, pre_noise_sga, eval_noise_sga,
                args.eval_mode, args.lam, args.burn_in_pct,
                not args.no_log_k,          # use_log_k (ignored when model_kwargs supplied)
                model=args.model,
                model_kwargs=model_kwargs,
            ) for x in solutions)
        es.tell(solutions, fitness_list)
        best_f = es.result.fbest
        print(f"Task {task_id} | Gen {gen:02d}: f_best={best_f:.6f}", flush=True)
        history.append(best_f)

    best_x = es.result.xbest
    print(f"Task {task_id} optimization completed. Best point: {best_x}")

    # --- Post-Optimization: Per-Point Weights (Point Estimate Mode Only) ---
    point_weights = None
    data_values   = None
    if args.mode == 'getPointEstimate':
        print("Computing per-point NPL weights...")
        _, g_final = original_sga_solve(
            best_x, X_np, pre_noise_sga, eval_noise_sga,
            args.lam, 0.0, model=args.model, model_kwargs=model_kwargs
        )
        v  = -args.lam * g_final
        v -= v.max()
        sigma_w  = np.exp(v)
        sigma_w /= sigma_w.sum()
        point_weights = sigma_w.tolist()   # length N; index i -> X_np[i]
        data_values   = X_np.tolist()      # raw data values, index-aligned
        print(f"Computed weights for {len(point_weights)} data points. "
              f"Min={min(point_weights):.6e}, Max={max(point_weights):.6e}")

    # --- Post-Optimization Diagnostics (Lambda Selection Only) ---
    w2_fit_reweighted = None
    if args.mode == 'lam_select' and ot is not None:
        print("Computing W2 diagnostic...")

        # 1. Final SGA run to get g
        _, g_final = original_sga_solve(
            best_x, X_np, pre_noise_sga, eval_noise_sga,
            args.lam, 0.0, model=args.model, model_kwargs=model_kwargs
        )

        # 2. Compute reweighted distribution
        v  = -args.lam * g_final
        v -= v.max()
        sigma_w  = np.exp(v)
        sigma_w /= sigma_w.sum()

        # 3. Sample from fitted model
        batch_size_w2 = 1000
        noise_w2 = torch.randn(batch_size_w2, 1)
        theta_t  = (torch.tensor(best_x, dtype=torch.float32)
                       .unsqueeze(0).expand(batch_size_w2, -1))

        if args.model == 'gandk':
            sim = GandKSimulator(d=1, **model_kwargs)
        else:  # normal
            sim = NormalMeanSimulator(d=1, **model_kwargs)

        with torch.no_grad():
            Z_fit = sim(theta_t, noise_w2).numpy().astype(np.float64)

        # 4. Compute W2^2 using POT
        M_dist   = (Z_fit - X_np.reshape(1, -1)) ** 2
        a_weights = np.ones(batch_size_w2) / batch_size_w2
        b_weights = sigma_w.astype(np.float64)
        b_weights /= b_weights.sum()

        try:
            w2_fit_reweighted = ot.emd2(a_weights, b_weights, M_dist,
                                         numItermax=1000000)
            print(f"Diagnostic W2^2: {w2_fit_reweighted:.6f}")
        except Exception as e:
            print(f"Error computing W2 diagnostic: {e}")

    end_time_exec  = time.time()
    total_exec_time = end_time_exec - start_time_exec
    print(f"Task {task_id} total execution time: {total_exec_time:.2f} seconds",
          flush=True)

    # --- Save Results ---
    result = {
        "task_id":           int(task_id),
        "mode":              str(args.mode),
        "model":             str(args.model),
        "dataset_id":        int(dataset_index),
        "lambda":            float(args.lam),
        "theta_est":         [float(x) for x in best_x],
        "final_f":           float(es.result.fbest),
        "w2_fit_reweighted": float(w2_fit_reweighted) if w2_fit_reweighted is not None else None,
        # getPointEstimate fields: parallel lists (index i -> data_values[i], point_weights[i])
        "data_values":       data_values,
        "point_weights":     point_weights,
        "params":            vars(args),
    }

    output_filename = f"task_{task_id}.json"
    with open(output_filename, 'w') as f:
        json.dump(result, f, indent=4)

    if args.bucket:
        if args.mode == 'lam_select':
            if args.model == 'normal':
                if args.NormalContamOption is None:
                    raise ValueError(
                        "--NormalContamOption (A, B, or C) is required when "
                        "--model normal and --mode lam_select"
                    )
                prefix = (f"{args.model}/results/N{args.N_data}/lamSelectResults"
                          f"/contamOption={args.NormalContamOption}")
            else:
                # gandk: no contamOption sub-directory
                prefix = f"{args.model}/results/N{args.N_data}/lamSelectResults"
        elif args.mode == 'getPointEstimate':
            if args.NormalContamOption is None:
                raise ValueError(
                    "--NormalContamOption (A, B, or C) is required when "
                    "--mode getPointEstimate"
                )
            prefix = (f"{args.model}/results/N{args.N_data}/pointEstResults"
                      f"/contamOption={args.NormalContamOption}")
        else:
            prefix = args.results_prefix

        gcs_tgt = f"{prefix}/{output_filename}"
        print(f"Uploading result to gs://{args.bucket}/{gcs_tgt}...")
        upload_to_gcs(output_filename, args.bucket, gcs_tgt)

    # Cleanup temp data file
    try:
        if ('local_data_path' in locals()
                and os.path.exists(local_data_path)
                and args.ground_truth_data.startswith("gs://")):
            os.remove(local_data_path)
    except Exception as e:
        print(f"Cleanup warning: {e}")


if __name__ == "__main__":
    main()
