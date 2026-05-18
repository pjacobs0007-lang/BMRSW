import numpy as np

def sample_ddm_groundTruth(a, tnd, w, v, N=1, dt=0.001, maxIter=10000):
    """
    Simulates N trials of the Simple Drift Diffusion Model (DDM).
    
    Parameters:
    a (float): boundary (decision boundaries at -a and a, a > 0)
    tnd (float): non-decision time (tnd > 0)
    w (float): initial position (-a < w < a)
    v (float): drift coefficient
    N (int): number of iid draws
    dt (float): time step for simulation
    maxIter (int): maximum number of iterations
    
    Returns:
    np.ndarray: Array of shape (2, N) containing reaction times in row 0 and choices in row 1.
                Choice is 1 if upper boundary 'a' is crossed, -1 if '-a' is crossed.
    """
    X = np.full(N, w, dtype=float)
    rt = np.full(N, np.nan)
    choice = np.zeros(N)
    
    active = np.ones(N, dtype=bool)
    sqrt_dt = np.sqrt(dt)
    
    for i in range(maxIter):
        if not np.any(active):
            break
            
        hit_upper = active & (X >= a)
        hit_lower = active & (X <= -a)
        
        current_time = tnd + i * dt
        rt[hit_upper] = current_time
        choice[hit_upper] = 1.0
        
        rt[hit_lower] = current_time
        choice[hit_lower] = -1.0
        
        active = active & ~(hit_upper | hit_lower)
        
        if not np.any(active):
            break
            
        num_active = np.sum(active)
        Z = np.random.normal(0, 1, size=num_active)
        X[active] += v * dt + sqrt_dt * Z
        
    return np.vstack((rt, choice))

def sample_clf_groundTruth(a, tnd, w, v, N=1, dt=0.001, maxIter=10000):
    """
    Simulates N trials of the Cauchy Levy Flight Model (CLF).
    
    Parameters:
    a (float): boundary (decision boundaries at -a and a, a > 0)
    tnd (float): non-decision time (tnd > 0)
    w (float): initial position (-a < w < a)
    v (float): drift coefficient
    N (int): number of iid draws
    dt (float): time step for simulation
    maxIter (int): maximum number of iterations
    
    Returns:
    np.ndarray: Array of shape (2, N) containing reaction times in row 0 and choices in row 1.
                Choice is 1 if upper boundary 'a' is crossed, -1 if '-a' is crossed.
    """
    X = np.full(N, w, dtype=float)
    rt = np.full(N, np.nan)
    choice = np.zeros(N)
    
    active = np.ones(N, dtype=bool)
    
    for i in range(maxIter):
        if not np.any(active):
            break
            
        hit_upper = active & (X >= a)
        hit_lower = active & (X <= -a)
        
        current_time = tnd + i * dt
        rt[hit_upper] = current_time
        choice[hit_upper] = 1.0
        
        rt[hit_lower] = current_time
        choice[hit_lower] = -1.0
        
        active = active & ~(hit_upper | hit_lower)
        
        if not np.any(active):
            break
            
        num_active = np.sum(active)
        C = np.random.standard_cauchy(size=num_active)
        X[active] += v * dt + dt * C
        
    return np.vstack((rt, choice))

def sample_gandk_groundTruth(a, b, g, k_param, eps, z, rho, N, use_log_k=True):
    """
    Generates a dataset of size N according to:
    X = (1-eps) * gandk(a,b,g,log_k) + eps * delta_{z}
    Y = floor(X / rho) * rho (if rho is specified and > 0)
    
    Y is the observed quantity.
    Returns an array of shape (N, 1)
    """
    k = np.exp(k_param) if use_log_k else k_param
    c = 0.8  # Standard g-and-k default
    
    # Generate mixture indices
    # 1 if outlier, 0 if gandk
    is_outlier = np.random.rand(N) < eps
    
    # Generate N standard normals
    Z = np.random.randn(N)
    
    # Calculate G-and-K
    term1 = 1 + c * np.tanh(g * Z / 2.0)
    term2 = (1 + Z**2)**k
    gandk_samples = a + b * term1 * term2 * Z
    
    # Combine according to mask
    X = np.where(is_outlier, z, gandk_samples)
    
    # Discretization
    if rho is not None and rho > 0:
        Y = np.floor(X / rho) * rho
    else:
        Y = X
        
    return Y.reshape(-1, 1)

def sample_normal_groundTruth(mu, sigma, N,
                               contamOption='C',
                               eps=0.0, nu=1.0, z=0.0, rho=None):
    """
    Generates a dataset of size N from a Normal(mu, sigma) base with optional
    contamination, controlled by contamOption.

    Parameters
    ----------
    mu           : float -- mean of the base normal distribution
    sigma        : float -- standard deviation of the base normal distribution
    N            : int   -- number of samples
    contamOption : str   -- one of 'A', 'B', 'C' (default 'C')
        'A' : Huber point-mass contamination.
              Each observation is drawn from:
                (1 - eps) * t_{nu}(mu, sigma)  +  eps * delta_{z}
              where t_{nu}(mu, sigma) is a t-distribution with `nu` degrees of
              freedom, location mu, and scale sigma.
        'B' : Mixture with a shifted-normal outlier component + discretization.
              With probability (1-eps): X ~ Normal(0,1), Y = floor(X/rho)*rho.
              With probability eps:     Y ~ Normal(8, 1).
              (mu, sigma ignored; rho must be provided and > 0)
        'C' : Clean normal.  Y ~ Normal(0, 1).
              (mu, sigma, eps, nu, z, rho all ignored)
    eps  : float -- contamination fraction in [0, 1) (used by A and B)
    nu   : float -- degrees of freedom for the t-distribution (used by A)
    z    : float -- point-mass location for delta contamination (used by A)
    rho  : float or None -- discretization step size (used by B)

    Returns
    -------
    numpy array of shape (N, 1), matching the output format of
    sample_gandk_groundTruth so the result can be used in the same
    downstream code paths.
    """
    from scipy.stats import t as t_dist

    if contamOption == 'A':
        # ------------------------------------------------------------------
        # Option A: (1 - eps) * t_{nu}(mu, sigma) + eps * delta_{z}
        # ------------------------------------------------------------------
        is_outlier = np.random.rand(N) < eps
        # t-distribution with nu df, location mu, scale sigma
        clean_samples = t_dist.rvs(df=nu, loc=mu, scale=sigma, size=N)
        Y = np.where(is_outlier, z, clean_samples)

    elif contamOption == 'B':
        # ------------------------------------------------------------------
        # Option B: discretized normal + shifted-normal outlier component
        # ------------------------------------------------------------------
        if rho is None or rho <= 0:
            raise ValueError(
                "contamOption='B' requires rho to be a positive float."
            )
        is_outlier = np.random.rand(N) < eps
        # Clean component: Normal(0,1) discretized at step rho
        X_clean = np.random.normal(loc=mu, scale=sigma, size=N)
        X_clean_disc = np.floor(X_clean / rho) * rho
        # Outlier component: Normal(8, 1)
        X_outlier = np.random.normal(loc=8.0, scale=1.0, size=N)
        Y = np.where(is_outlier, X_outlier, X_clean_disc)

    elif contamOption == 'C':
        # ------------------------------------------------------------------
        # Option C: clean Normal(0, 1)
        # ------------------------------------------------------------------
        Y = np.random.normal(loc=mu, scale=sigma, size=N)

    else:
        raise ValueError(
            f"contamOption must be 'A', 'B', or 'C'; got '{contamOption}'."
        )

    return Y.reshape(-1, 1)

def sample_SLR_groundTruth(b0, b1, sigma, N,
                           contamOption='A', eps=0.0,
                           b0Alt=0.0, b1Alt=0.0,
                           l=-1.0, u=1.0, shift=0.0):
    """
    Generates a dataset of size N for Simple Linear Regression (SLR) with 
    optional contamination and local shifts.

    Parameters
    ----------
    b0, b1  : float -- Base intercept and slope
    sigma   : float -- Noise standard deviation
    N       : int   -- Number of samples
    contamOption : str -- 'A' or 'C'
        'A' : Mixture of a 'Clean' component with a local shift and 
              an 'Outlier' component with a different regression line.
        'C' : Standard clean SLR: Y = b0 + b1*X + sigma*eps
    eps     : float -- Fraction of outliers (Option A)
    b0Alt, b1Alt : float -- Alternative intercept/slope for outliers
    l, u    : float -- Bounds for local shift (Clean component in Option A)
    shift   : float -- Additive shift applied to Y if X in (l, u)

    Returns
    -------
    np.ndarray : (N, 2) array where col 0 is Y and col 1 is X.
    """
    X = np.random.uniform(-1, 1, size=N)
    
    if contamOption == 'A':
        # Mixture logic
        is_outlier = np.random.rand(N) < eps
        
        # Clean component: b0 + b1*X + shift*I(X in (l,u)) + Noise
        in_interval = (X > l) & (X < u)
        mean_clean = b0 + b1 * X + shift * in_interval
        
        # Outlier component: b0Alt + b1Alt*X + Noise
        mean_outlier = b0Alt + b1Alt * X
        
        mean = np.where(is_outlier, mean_outlier, mean_clean)
        Y = mean + np.random.normal(0, sigma, size=N)
        
    elif contamOption == 'C':
        # Standard clean SLR
        Y = b0 + b1 * X + np.random.normal(0, sigma, size=N)
    else:
        raise ValueError(f"Unknown contamOption '{contamOption}' for SLR model.")

    return np.column_stack((Y, X))

def compute_w2_and_entropy(g_final, lam, inference_simulator, theta_final, X_train, device='cpu'):
    """
    Computes W2^2(F_theta*, L) and entropy of the reweighting.
    L is the reweighted discrete distribution on X_train induced by g_final and lam.
    """
    import torch
    import ot
    
    with torch.no_grad():
        # 1. Weights L (p_i)
        if np.isinf(lam):
            p = torch.softmax(-1e6 * g_final, dim=0)
        else:
            p = torch.softmax(-lam * g_final, dim=0)
            
        entropy = -torch.sum(p * torch.log(p + 1e-12)).item()
        
        # 2. Sample from model F_theta*
        # Ensure theta_final is a tensor
        if not isinstance(theta_final, torch.Tensor):
            theta_final = torch.tensor(theta_final, dtype=torch.float32, device=device)
        
        d = X_train.shape[1]
        batch_size_w2 = 1000
        noise_w2 = torch.randn(batch_size_w2, d, device=device)
        theta_batch_final = theta_final.unsqueeze(0).expand(batch_size_w2, -1)
        Z_final = inference_simulator(theta_batch_final, noise_w2)
        
        # 3. Compute Distance Matrix
        M = torch.cdist(Z_final, X_train, p=2)**2
        
        # 4. Compute W2^2 using ot.emd2
        a = np.ones(batch_size_w2) / batch_size_w2
        b = p.detach().cpu().numpy().astype(np.float64)
        b = b / b.sum()
        
        M_np = M.detach().cpu().numpy().astype(np.float64)
        
        try:
            w2_val = ot.emd2(a, b, M_np, numItermax=1000000)
        except Exception as e:
            print(f"Error computing W2: {e}")
            w2_val = np.nan
            
    return entropy, w2_val, b

def compute_bias_variance(theta_ref_np, all_finals):
    """
    Computes bias-variance decomposition for a set of bootstrap estimates.
    theta_ref_np: (n_params,) array of the reference point (e.g. Ground Truth)
    all_finals: (n_samples, n_params) array of bootstrap estimates
    """
    mean_estimate = np.mean(all_finals, axis=0)  # E[theta_hat]
    
    # Variance: E[(theta_hat - E[theta_hat])^2]
    variances = np.mean((all_finals - mean_estimate)**2, axis=0)
    
    # Squared Bias: (E[theta_hat] - theta_ref)^2
    sq_bias = (mean_estimate - theta_ref_np)**2
    
    # Total MSE = Variance + Bias^2
    approx_mse = variances + sq_bias
    avg_joint_mse = float(np.mean(approx_mse))
    
    return variances, sq_bias, approx_mse, avg_joint_mse

def plot_lam_select_metrics(summary_data, all_entropy_vals, all_w2_vals, labels, theta_true, 
                            results_dir, args_lam, uncertainty_mode, model_name,
                            all_weight_data=None, d=1):
    """
    Renders the unified metrics grid for lambda selection.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import numpy as np
    
    n_params = len(labels)
    has_mse = theta_true is not None
    n_lambdas = len(args_lam)
    
    n_plot_rows = 2 # Entropy, W2
    if all_weight_data is not None and (d == 1 or model_name in ['regression', 'meanFreq', 'smoothStep', 'gandk']):
        n_plot_rows += n_lambdas
    if has_mse:
        n_plot_rows += 1
    
    # Check if we have variance/bias data (only in lam_select mode)
    has_var_bias = len(summary_data) > 0 and summary_data[0].get('variances') is not None
    
    if has_var_bias:
        n_plot_rows += 4 # variance, sq_bias, approx_mse (3) + avg_joint_mse (1)
        
    num_params_to_plot = n_params
    n_cols = num_params_to_plot
    fig_width = max(10, 4 * n_cols)
    fig_height = max(10, 4 * n_plot_rows) 
    fig_ent = plt.figure(figsize=(fig_width, fig_height))
    gs = fig_ent.add_gridspec(n_plot_rows, n_cols)
    
    row_idx = 0
    formatted_lams = [f"{l:.2g}" for l in args_lam]
    x_positions = range(len(args_lam))

    # 1. Parameter MSE Plot (Bias Inspection)
    if has_mse:
        for i in range(num_params_to_plot):
            ax_p = fig_ent.add_subplot(gs[row_idx, i])
            true_val = theta_true[i]
            
            mse_vals = []
            for d_sum in summary_data:
                finals = d_sum['all_finals'] 
                if finals.shape[1] > i:
                    mse = np.mean((finals[:, i] - true_val)**2)
                else:
                    mse = 0.0
                mse_vals.append(mse)
                
            color = plt.cm.tab10(i % 10)
            ax_p.plot(x_positions, mse_vals, marker='o', linestyle='-', color=color)
            ax_p.set_xticks(x_positions)
            ax_p.set_xticklabels(formatted_lams, rotation=45)
            ax_p.set_xlabel(r'$\lambda$')
            ax_p.set_ylabel(f'MSE vs {labels[i]} ({true_val:.2f})')
            ax_p.set_title(f'Bootstrap MSE of {labels[i]}')
            if np.any(np.array(mse_vals) > 1e-12):
                ax_p.set_yscale('log')
            ax_p.grid(True, which='both', alpha=0.3)
        row_idx += 1
            
    # 2. Variance, Bias^2, Approx MSE Plots
    def _plot_per_param_row(metric_key, title_prefix, ylabel_prefix, color, row_i):
        for pi in range(n_cols):
            ax = fig_ent.add_subplot(gs[row_i, pi])
            vals = [d_sum[metric_key][pi] for d_sum in summary_data]
            ax.plot(x_positions, vals, marker='o', linestyle='-', color=color)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(formatted_lams, rotation=45)
            ax.set_xlabel(r'$\lambda$')
            ax.set_ylabel(ylabel_prefix)
            ax.set_title(f'{title_prefix} – {labels[pi]}')
            if np.any(np.array(vals) > 1e-12):
                ax.set_yscale('log')
            ax.grid(True, which='both', alpha=0.3)

    if has_var_bias:
        _plot_per_param_row('variances',   'Estimator Variance',     'Variance',     'tab:red',    row_idx); row_idx += 1
        _plot_per_param_row('sq_bias',    'Estimator Squared Bias', 'Squared Bias', 'tab:orange', row_idx); row_idx += 1
        _plot_per_param_row('approx_mse', 'Estimator Approx MSE',   'Approx MSE',   'tab:green',  row_idx); row_idx += 1

        # Average Joint MSE
        ax_joint = fig_ent.add_subplot(gs[row_idx, :])
        row_idx += 1
        joint_vals = [d_sum['avg_joint_mse'] for d_sum in summary_data]
        ax_joint.plot(x_positions, joint_vals, marker='o', linestyle='-', color='tab:purple')
        ax_joint.set_xticks(x_positions)
        ax_joint.set_xticklabels(formatted_lams, rotation=45)
        ax_joint.set_xlabel(r'$\lambda$')
        ax_joint.set_ylabel(r'Avg Joint Approx MSE')
        ax_joint.set_title(r'Average Joint Approx MSE vs. $\lambda$')
        if np.any(np.array(joint_vals) > 1e-12):
            ax_joint.set_yscale('log')
        ax_joint.grid(True, which='both', alpha=0.3)

    # 3. Entropy Plot
    ax_ent = fig_ent.add_subplot(gs[row_idx, :])
    row_idx += 1
    sns.boxplot(data=all_entropy_vals, ax=ax_ent, palette="viridis")
    ax_ent.set_xticks(range(len(args_lam)))
    ax_ent.set_xticklabels(formatted_lams, rotation=45)
    ax_ent.set_xlabel(r'$\lambda$')
    ax_ent.set_ylabel(r'Entropy ($-\sum p_j \log p_j$)')
    ax_ent.set_title(f'Entropy vs. $\lambda$ ({uncertainty_mode})')
    ax_ent.grid(True, alpha=0.3)
    
    # 4. W2^2 Plot
    ax_w2 = fig_ent.add_subplot(gs[row_idx, :])
    row_idx += 1
    sns.boxplot(data=all_w2_vals, ax=ax_w2, palette="magma")
    ax_w2.set_xticks(range(len(args_lam)))
    ax_w2.set_xticklabels(formatted_lams, rotation=45)
    ax_w2.set_xlabel(r'$\lambda$')
    ax_w2.set_ylabel(r'$W_2^2(F_{\theta^*}, L)$ (Log Scale)')
    ax_w2.set_title(r'$W_2^2$ Distance vs. $\lambda$')
    ax_w2.set_yscale('log')
    ax_w2.grid(True, which='both', alpha=0.3)
    
    # 5. Weights Debug Plots
    if all_weight_data is not None:
        for i, (lam_val, weight_info) in enumerate(zip(args_lam, all_weight_data)):
            ax_weight = fig_ent.add_subplot(gs[row_idx, :])
            row_idx += 1
            Xs_list, ps_list = weight_info
            
            for j in range(len(Xs_list)):
                X_samples = Xs_list[j]
                p_w = ps_list[j].flatten()
                
                if model_name in ['regression', 'meanFreq'] and X_samples.ndim == 2 and X_samples.shape[1] == 2:
                    Y_vals = X_samples[:, 0]
                    X_vals = X_samples[:, 1]
                    ax_weight.scatter(X_vals, Y_vals, c=p_w, cmap='viridis', s=10, alpha=0.4)
                else:
                    X_w = X_samples.flatten()
                    sort_idx = np.argsort(X_w)
                    ax_weight.plot(X_w[sort_idx], p_w[sort_idx], alpha=0.5, linewidth=1)
            
            ax_weight.set_title(f'Weights vs. Data for $\lambda={lam_val}$')
            ax_weight.grid(True, alpha=0.3)
    
    try:
        fig_ent.tight_layout()
    except Exception:
        fig_ent.subplots_adjust(hspace=0.5, wspace=0.3)
    
    ent_path = os.path.join(results_dir, "entropy.png")
    fig_ent.savefig(ent_path)
    plt.close(fig_ent)
    print(f"Combined Metrics Grid plot saved to '{ent_path}'")

def sample_predatorPreyInteraction(w, r, drift_angle, drift_mag, N=1, dt=0.001, maxIter=10000):
    """
    Simulates a Predator-Prey 2D Drift-Diffusion process using Euler-Maruyama.
    
    Predator starts at (0, 0).
    Prey starts at w (a 2D vector), with |w| < 1 and |w| > r.
    Both move according to 2D Brownian motion with drift (determined by drift_angle and drift_mag).
    They have independent driving noise but the same drift parameters.
    
    Stopping conditions:
    1. Caught: |Predator(t) - Prey(t)| <= r. Final output is Predator(t).
    2. Survives: |Prey(t)| >= 1. Final output is Prey(t).
    
    Args:
        w (tuple or np.ndarray): Initial position of prey (x, y).
        r (float): Catch radius of predator.
        drift_angle (float): Angle of the drift vector (in radians).
        drift_mag (float): Magnitude of the drift vector.
        N (int): Number of parallel simulations.
        dt (float): Time step.
        maxIter (int): Maximum iterations to prevent infinite loops.
        
    Returns:
        np.ndarray: Final positions of the prey (or predator if caught), shape (2, N).
    """
    w = np.array(w, dtype=float)
    drift = np.array([np.cos(drift_angle), np.sin(drift_angle)]) * drift_mag
    
    # Shapes will be (2, N)
    pos_pred = np.zeros((2, N))
    pos_prey = np.tile(w[:, None], (1, N))
    
    final_pos = np.full((2, N), np.nan)
    survived_mask = np.zeros(N, dtype=bool)
    active = np.ones(N, dtype=bool)
    
    sqrt_dt = np.sqrt(dt)
    drift_step = drift[:, None] * dt
    
    for _ in range(maxIter):
        if not np.any(active):
            break
            
        n_active = np.sum(active)
        
        # Independent noise for both predator and prey
        noise_pred = np.random.randn(2, n_active) * sqrt_dt
        noise_prey = np.random.randn(2, n_active) * sqrt_dt
        
        # Euler-Maruyama update
        pos_pred[:, active] += drift_step + noise_pred
        pos_prey[:, active] += drift_step + noise_prey
        
        # Check stopping conditions for active walkers
        # 1. Caught: Distance between predator and prey <= r
        diff = pos_pred[:, active] - pos_prey[:, active]
        dist_between = np.sqrt(np.sum(diff**2, axis=0))
        caught = dist_between <= r
        
        # 2. Survived: Prey hits the unit circle boundary
        dist_prey_origin = np.sqrt(np.sum(pos_prey[:, active]**2, axis=0))
        survived = dist_prey_origin >= 1.0
        
        finished = caught | survived
        
        if np.any(finished):
            # Map back to the original indices
            active_indices = np.where(active)[0]
            finished_indices = active_indices[finished]
            
            # Extract the boolean masks relative to the finished subset
            caught_finished = caught[finished]
            
            # Update final positions
            for i, idx in enumerate(finished_indices):
                if caught_finished[i]:
                    final_pos[:, idx] = pos_pred[:, idx]
                    survived_mask[idx] = False
                else:
                    final_pos[:, idx] = pos_prey[:, idx]
                    survived_mask[idx] = True
                    
            # Mark as inactive
            active[finished_indices] = False
            
    # For any that did not finish within maxIter, assign their last known position
    if np.any(active):
        final_pos[:, active] = pos_prey[:, active]
        survived_mask[active] = True
        
    return final_pos, survived_mask
