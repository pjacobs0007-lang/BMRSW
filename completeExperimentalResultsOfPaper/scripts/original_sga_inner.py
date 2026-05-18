"""
original_sga_inner.py

Original SGA solver for the semi-dual objective.
Targets the weighted average of objective values along the trajectory.
"""

import numpy as np
import torch
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None

def _soft_assignments(g, dists_LN, eps):
    if eps > 0:
        scores = (g[None, :] - dists_LN) / eps
        scores -= scores.max(axis=1, keepdims=True)
        e = np.exp(scores)
        return e / e.sum(axis=1, keepdims=True)
    else:
        M = dists_LN - g[None, :]
        min_indices = np.argmin(M, axis=1)
        P = np.zeros_like(M)
        P[np.arange(len(M)), min_indices] = 1.0
        return P

def _softmax_sigma(g, lam):
    v = -lam * g - (-lam * g).max()
    e = np.exp(v)
    return e / e.sum()

def _eval_objective(g, dists_LN, lam, eps):
    N = len(g)
    if eps > 0:
        exponents = (g[None, :] - dists_LN) / eps
        shift     = exponents.max(axis=1, keepdims=True)
        log_sum   = np.log(np.exp(exponents - shift).sum(axis=1)) + shift.ravel()
        first_term = (-eps * log_sum).mean()
    else:
        first_term = np.min(dists_LN - g[None, :], axis=1).mean()

    v = -lam * g
    max_v = v.max()
    second_term = -(1.0 / lam) * (np.log(np.exp(v - max_v).sum()) + max_v - np.log(N))
    return first_term + second_term

def original_sga_solve_slow(theta_np, X_np, pre_fixed_noises, eval_noise_np,
                        lam, soft_min_eps=0.0, model='gandk'):
    """Original (slow) SGA solver using per-iteration simulator calls."""
    from differentiable_simulators import GandKSimulator, NormalMeanSimulator, MeanFreqSimulator, InterceptFreqSimulator, UnivariateMeanFreqSimulator, DiscretizedNormalSimulator, SmoothStepSimulator
    if model == 'gandk':
        sim = GandKSimulator(d=1)
    elif model == 'normal':
        sim = NormalMeanSimulator(d=1, sigma_fixed=None, parameterization='linear')
    elif model == 'meanFreq':
        sim = MeanFreqSimulator()
    elif model == 'interceptFreq':
        sim = InterceptFreqSimulator()
    elif model == 'univariateMeanFreq':
        sim = UnivariateMeanFreqSimulator()
    elif model == 'discretizedNormal':
        sim = DiscretizedNormalSimulator()
    elif model == 'smoothStep':
        sim = SmoothStepSimulator()
    else:
        raise ValueError(f"Unknown model: {model}")

    N    = X_np.shape[0]
    X_val = X_np.astype(np.float64) # Keep shape (N, d)
    theta_t = torch.tensor(theta_np, dtype=torch.float32)

    g = np.zeros(N, dtype=np.float64)
    
    f_total = 0.0
    sum_lr  = 0.0

    for i, noise_np in enumerate(pre_fixed_noises, 1):
        # 1. Learning Rate: sqrt(N/t)
        lr = np.sqrt(N / i)
        
        # 2. Simulate
        L = noise_np.shape[0]
        noise = torch.tensor(noise_np, dtype=torch.float32)
        tb    = theta_t.unsqueeze(0).expand(L, -1)
        with torch.no_grad():
            Z = sim(tb, noise).numpy().astype(np.float64)
        
        if Z.ndim == 1:
            Z = Z[:, None]
        if X_val.ndim == 1:
            X_val_2d = X_val[:, None]
        else:
            X_val_2d = X_val
            
        dists_LN = ((Z[:, None, :] - X_val_2d[None, :, :]) ** 2).sum(axis=-1)
        
        # 3. Objective Estimate O_t
        obj_t = _eval_objective(g, dists_LN, lam, soft_min_eps)
        f_total += lr * obj_t
        sum_lr  += lr
        
        # 4. Gradient Step
        P     = _soft_assignments(g, dists_LN, soft_min_eps)
        sigma = _softmax_sigma(g, lam)
        p_bar = P.mean(axis=0)
        grad  = -p_bar + sigma  # SGA maximizes
        
        g = g + lr * grad
        
    return f_total / sum_lr, g


def original_sga_solve(theta_np, X_np, pre_fixed_noises, eval_noise_np,
                       lam, soft_min_eps=0.0, model='gandk', model_kwargs=None, burn_in_pct=0.0):
    """
    SGA solver using trajectory averaging for the objective value.
    This version batches the simulator in Torch but performs the loop in NumPy or JAX.
    """
    if model_kwargs is None:
        model_kwargs = {}
        
    from differentiable_simulators import GandKSimulator, NormalMeanSimulator, MeanFreqSimulator, InterceptFreqSimulator, UnivariateMeanFreqSimulator, DiscretizedNormalSimulator, SmoothStepSimulator
    
    # 1. Simulator setup
    if model == 'gandk':
        sim = GandKSimulator(d=1, **model_kwargs)
    elif model == 'normal':
        sim = NormalMeanSimulator(d=1, **model_kwargs)
    else:
        # Fallback for other models (less optimized)
        sim_map = {'meanFreq': MeanFreqSimulator, 'interceptFreq': InterceptFreqSimulator, 
                   'univariateMeanFreq': UnivariateMeanFreqSimulator, 'discretizedNormal': DiscretizedNormalSimulator, 
                   'smoothStep': SmoothStepSimulator}
        sim = sim_map[model](**model_kwargs)

    N = X_np.shape[0]
    X_val = X_np.astype(np.float64)
    theta_t = torch.tensor(theta_np, dtype=torch.float32)
    
    # 2. Batch Simulator for all iterations (In Torch)
    if isinstance(pre_fixed_noises, list):
        noise_full_np = np.concatenate(pre_fixed_noises)
    else:
        noise_full_np = pre_fixed_noises
        
    noise_full = torch.tensor(noise_full_np, dtype=torch.float32)
    total_L = noise_full.shape[0]
    T = total_L # Assuming 1 noise sample per SGA iteration (L_per_iter=1)
    L_per_iter = 1
    
    if total_L > 20000: # Sanity check if batching is used
         L_per_iter = total_L // 20000 # This logic is fragile, better to pass L_per_iter
         T = 20000

    with torch.no_grad():
        tb_full = theta_t.unsqueeze(0).expand(total_L, -1)
        Z_full = sim(tb_full, noise_full).numpy().astype(np.float64)
        if Z_full.ndim == 1:
            Z_full = Z_full[:, None]
        if X_val.ndim == 1:
            X_val = X_val[:, None]

    burn_in_steps = int(T * (burn_in_pct / 100.0))

    # 3. Fast JAX Path if available
    if jax is not None:
        return _jax_sga_solve_internal(Z_full, X_val, lam, soft_min_eps, T, L_per_iter, burn_in_steps)
    
    # 4. Sequential Loop in NumPy (Optimized)
    g = np.zeros(N, dtype=np.float64)
    f_total = 0.0
    sum_lr  = 0.0
    
    X_val_ravel = X_val.ravel()
    inv_lam = 1.0 / lam
    log_N = np.log(N)
    
    for i in range(1, T + 1):
        lr = np.sqrt(N / i)
        Z_curr = Z_full[(i-1)*L_per_iter : i*L_per_iter].ravel() # Optimizing for d=1
        
        # d=1 Distance
        dists_N = (Z_curr[:, None] - X_val_ravel[None, :])**2 
        
        # Softmax Sigma for dual variable
        v = -lam * g
        v_max = v.max()
        e_sig = np.exp(v - v_max)
        e_sig_sum = e_sig.sum()
        sigma = e_sig / e_sig_sum
        
        # Objective Term O_t
        if i > burn_in_steps:
            if soft_min_eps > 0:
                exponents = (g[None, :] - dists_N) / soft_min_eps
                shift     = exponents.max(axis=1, keepdims=True)
                log_sum   = np.log(np.exp(exponents - shift).sum(axis=1)) + shift.ravel()
                first_term = (-soft_min_eps * log_sum).mean()
            else:
                first_term = np.min(dists_N - g[None, :], axis=1).mean()
            
            second_term = -inv_lam * (np.log(e_sig_sum) + v_max - log_N)
            f_total += lr * (first_term + second_term)
            sum_lr  += lr
            
        # Gradient
        if soft_min_eps > 0:
            scores = (g[None, :] - dists_N) / soft_min_eps
            scores -= scores.max(axis=1, keepdims=True)
            e_p = np.exp(scores)
            p_bar = (e_p / e_p.sum(axis=1, keepdims=True)).mean(axis=0)
        else:
            min_idx = np.argmin(dists_N - g[None, :], axis=1)
            # Fast p_bar for L=1
            if L_per_iter == 1:
                p_bar = np.zeros(N)
                p_bar[min_idx[0]] = 1.0
            else:
                p_bar = np.zeros(N)
                for idx in min_idx: p_bar[idx] += 1.0
                p_bar /= L_per_iter
                
        g += lr * (-p_bar + sigma)

    return (f_total / sum_lr if sum_lr > 0 else 0.0), g

def _jax_sga_solve_internal(Z_full_np, X_val_np, lam, soft_min_eps, T, L, burn_in_steps):
    N = X_val_np.shape[0]
    Z_full = jnp.array(Z_full_np)
    X_val = jnp.array(X_val_np).ravel()
    
    @jax.jit
    def body_fun(carry, i):
        g, f_total, sum_lr = carry
        lr = jnp.sqrt(N / (i + 1))
        
        Z_curr = jax.lax.dynamic_slice_in_dim(Z_full, i * L, L).ravel()
        dists_N = (Z_curr[:, None] - X_val[None, :])**2
        
        # Dual distribution sigma
        sigma = jax.nn.softmax(-lam * g)
        
        # Assignment distribution p_bar
        if soft_min_eps > 0:
            p_bar = jax.nn.softmax((g[None, :] - dists_N) / soft_min_eps, axis=1).mean(axis=0)
        else:
            min_idx = jnp.argmin(dists_N - g[None, :], axis=1)
            p_bar = jax.nn.one_hot(min_idx, N).mean(axis=0)
            
        new_g = g + lr * (-p_bar + sigma)
        
        # Objective
        is_post_burn = i >= burn_in_steps
        def compute_obj():
            if soft_min_eps > 0:
                first_term = -soft_min_eps * jax.nn.logsumexp((g[None, :] - dists_N) / soft_min_eps, axis=1).mean()
            else:
                first_term = jnp.min(dists_N - g[None, :], axis=1).mean()
            second_term = -(1.0 / lam) * (jax.nn.logsumexp(-lam * g) - jnp.log(N))
            return first_term + second_term

        # Only compute objective if needed to save time (though JIT might do this anyway)
        obj_t = jax.lax.cond(is_post_burn, compute_obj, lambda: 0.0)
        new_f_total = f_total + jax.lax.select(is_post_burn, lr * obj_t, 0.0)
        new_sum_lr = sum_lr + jax.lax.select(is_post_burn, lr, 0.0)
        
        return (new_g, new_f_total, new_sum_lr), None

    g_init = jnp.zeros(N)
    (g_final, f_total, sum_lr), _ = jax.lax.scan(body_fun, (g_init, 0.0, 0.0), jnp.arange(T))
    
    return float(f_total / sum_lr), np.array(g_final)
