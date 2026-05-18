import numpy as np
import torch
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None

from simulators import GandKSimulator, NormalMeanSimulator

def original_sga_solve(theta_np, X_np, pre_fixed_noises,
                       lam, soft_min_eps=0.0, model='gandk', model_kwargs=None, burn_in_pct=0.0):
    """
    SGA solver using trajectory averaging for the objective value.
    This version batches the simulator in Torch but performs the loop in NumPy or JAX.
    """
    if model_kwargs is None:
        model_kwargs = {}
        
    # 1. Simulator setup
    if model == 'gandk':
        sim = GandKSimulator(d=1, **model_kwargs)
    elif model == 'normal':
        sim = NormalMeanSimulator(d=1, **model_kwargs)
    else:
        raise ValueError(f"Model {model} not supported in this simplified version. Only 'gandk' and 'normal' are available.")

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
    
    inv_lam = 1.0 / lam
    log_N = np.log(N)
    
    for i in range(1, T + 1):
        lr = np.sqrt(N / i)
        Z_curr = Z_full[(i-1)*L_per_iter : i*L_per_iter]
        
        # Distance calculation (General for d >= 1)
        if Z_curr.shape[1] == 1:
            dists_N = (Z_curr.ravel()[:, None] - X_val.ravel()[None, :])**2 
        else:
            dists_N = ((Z_curr[:, None, :] - X_val[None, :, :])**2).sum(axis=-1)
        
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
    X_val = jnp.array(X_val_np)
    
    @jax.jit
    def body_fun(carry, i):
        g, f_total, sum_lr = carry
        lr = jnp.sqrt(N / (i + 1))
        
        Z_curr = jax.lax.dynamic_slice_in_dim(Z_full, i * L, L)
        
        if Z_curr.shape[1] == 1:
            dists_N = (Z_curr.ravel()[:, None] - X_val.ravel()[None, :])**2
        else:
            dists_N = ((Z_curr[:, None, :] - X_val[None, :, :])**2).sum(axis=-1)
        
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
