"""
plot_lam_select.py
------------------
Two-panel (or three-panel for contamOption=A) lam_select diagnostic figure.

  Panel 1 — Boxplot of W2(F_theta_hat, Q_hat) per lambda
  Panel 2 — Total MSE to theta_true, averaged over bootstrap samples, vs lambda
  Panel 3 — (contamOption=A only) Density overlay:
               • Blue  : clean Normal(mu_true, sigma_true)
               • Red   : contaminated DGP  (1-eps)*t_nu + eps*delta_z
               • Thin  : fitted Normal(mu_hat, sigma_hat) for each bootstrap
                         at every lambda in --lamsForPlot

Usage examples
--------------
python plot_lam_select.py --model normal --N 1000 --contamOption A
python plot_lam_select.py --model normal --N 1000 --contamOption A \\
    --lamsForPlot 0.01 1.0 100 --eps 0.05 --nu 12 --z 6
python plot_lam_select.py --model gandk  --N 1000 --theta_true 3 1 2 0.5
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy.stats import norm, t as t_dist

from lambda_selection_diagnostic_viz import draw_w2_boxplot


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def build_results_path(local_bucket_root: str, model: str, N: int,
                       contamOption: Optional[str]) -> str:
    if model == 'normal':
        if contamOption is None:
            raise ValueError("--contamOption is required when --model normal")
        rel = f"results/{model}/contamOption{contamOption}/lamSelect"
    else:
        raise ValueError(f"Unknown model '{model}'.")
    return os.path.join(local_bucket_root, rel)


def build_point_est_path(local_bucket_root: str, model: str, N: int,
                         contamOption: str) -> str:
    rel = f"results/{model}/bmrsw/contamOption{contamOption}/singleRun"
    return os.path.join(local_bucket_root, rel)


def build_production_path(local_bucket_root: str, model: str, N: int,
                          contamOption: str) -> str:
    """Path to production (fixed-lambda bootstrap) results."""
    rel = f"results/{model}/bmrsw/contamOption{contamOption}/performUQ"
    return os.path.join(local_bucket_root, rel)


def build_wasser_path(local_bucket_root: str, model: str, N: int,
                      contamOption: str) -> str:
    """Path to Wasserstein-2 bootstrap results."""
    rel = f"results/{model}/wasser/contamOption{contamOption}"
    return os.path.join(local_bucket_root, rel)


def load_bootstrap_thetas(prod_dir: str) -> List[List[float]]:
    """
    Reads all task JSONs in prod_dir and returns a list of theta_est vectors
    (each a list of floats) for every successfully completed bootstrap run.
    """
    thetas = []
    if not os.path.exists(prod_dir):
        print(f"[WARNING] Production dir not found: {prod_dir}", file=sys.stderr)
        return thetas
    for fname in sorted(f for f in os.listdir(prod_dir) if f.endswith(".json")):
        try:
            with open(os.path.join(prod_dir, fname)) as fh:
                res = json.load(fh)
            theta_est = res.get("theta_est")
            if theta_est and len(theta_est) >= 2:
                thetas.append([float(v) for v in theta_est])
        except Exception as e:
            print(f"[WARNING] Skipping {fname}: {e}", file=sys.stderr)
    return thetas


def load_data(results_dir: str) -> pd.DataFrame:
    """Reads all task JSONs; returns DataFrame with lambda, w2, theta_0, theta_1, …"""
    if not os.path.exists(results_dir):
        print(f"[WARNING] Results dir not found: {results_dir}", file=sys.stderr)
        return pd.DataFrame()

    rows = []
    for fname in sorted(f for f in os.listdir(results_dir) if f.endswith(".json")):
        try:
            with open(os.path.join(results_dir, fname)) as fh:
                res = json.load(fh)
            lam       = res.get("lambda") or res.get("lambd")
            w2        = res.get("w2_fit_reweighted") or res.get("w_2_fit_reweighted")
            theta_est = res.get("theta_est")
            if lam is None:
                continue
            row = {"lambda": float(lam)}
            if w2 is not None:
                row["w2"] = float(w2)
            if theta_est:
                for i, v in enumerate(theta_est):
                    row[f"theta_{i}"] = float(v)
            rows.append(row)
        except Exception as e:
            print(f"[WARNING] Skipping {fname}: {e}", file=sys.stderr)
    return pd.DataFrame(rows)


def load_point_est_data(results_dir: str,
                        lams_for_plot: List[float]) -> Dict[float, Dict]:
    """
    Loads pointEstResults JSONs whose lambda is in lams_for_plot (snapped to
    the closest stored value).  Returns dict: {stored_lambda -> {data_values, point_weights}}.
    """
    if not os.path.exists(results_dir):
        print(f"[WARNING] Point-est dir not found: {results_dir}", file=sys.stderr)
        return {}

    # Gather all stored lambdas and their records
    records: Dict[float, Dict] = {}
    for fname in sorted(f for f in os.listdir(results_dir) if f.endswith(".json")):
        try:
            with open(os.path.join(results_dir, fname)) as fh:
                res = json.load(fh)
            lam    = res.get("lambda")
            dv     = res.get("data_values")
            pw     = res.get("point_weights")
            if lam is None or dv is None or pw is None:
                continue
            records[float(lam)] = {
                "data_values":   np.array(dv,  dtype=np.float64),
                "point_weights": np.array(pw, dtype=np.float64),
            }
        except Exception as e:
            print(f"[WARNING] Skipping {fname}: {e}", file=sys.stderr)

    if not records:
        return {}

    stored_lams = np.array(sorted(records.keys()))

    # Snap each requested lambda to the closest stored one
    out: Dict[float, Dict] = {}
    for lam_req in lams_for_plot:
        closest = float(stored_lams[np.argmin(np.abs(stored_lams - lam_req))])
        out[closest] = records[closest]
    return out


def _sorted_lam_labels(df: pd.DataFrame) -> List[str]:
    return (df[["lambda", "lambda_str"]].drop_duplicates()
            .sort_values("lambda")["lambda_str"].tolist())


def _param_names(model: str, n: int) -> List[str]:
    if model == "gandk" and n == 4:
        return ["a", "b", "g", "k"]
    if model == "normal" and n == 2:
        return [r"\mu", r"\sigma"]
    return [rf"\theta_{i}" for i in range(n)]




# ---------------------------------------------------------------------------
# Panel 2 — Total MSE curve
# ---------------------------------------------------------------------------
def _draw_mse_curve(ax, df: pd.DataFrame,
                    theta_true: List[float],
                    lam_order: List[str],
                    model: str, N: int,
                    contamOption: Optional[str]) -> None:
    theta_cols = sorted(c for c in df.columns if c.startswith("theta_"))
    n_params   = min(len(theta_cols), len(theta_true))

    if not theta_cols or not theta_true:
        ax.text(0.5, 0.5, "No theta_est data", ha="center", va="center",
                transform=ax.transAxes)
        return

    df = df.copy()
    for i in range(n_params):
        df[f"se_{i}"] = (df[f"theta_{i}"] - theta_true[i]) ** 2
    df["total_se"] = df[[f"se_{i}" for i in range(n_params)]].sum(axis=1)

    agg = (df.groupby("lambda", sort=True)["total_se"]
             .mean().reset_index())

    x_pos    = np.arange(len(agg))
    lam_strs = [f"{v:.2g}" for v in agg["lambda"].values]

    ax.plot(x_pos, agg["total_se"].values,
            marker="o", markersize=5, color="black", linewidth=2)
    ax.set_yscale("log")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(lam_strs, rotation=45, ha="right")
    ax.set_xlabel(r"$\lambda$", fontsize=16)
    ax.set_ylabel("Mean Total MSE (log scale)", fontsize=16)

    contam_label = f", contamOption={contamOption}" if (
        model == "normal" and contamOption) else ""
    ax.set_title(
        rf"Mean MSE to $\theta^*$ (model={model}, N={N}{contam_label})",
        fontsize=18)
    ax.grid(True, alpha=0.3, which="both")
    ax.tick_params(axis="both", labelsize=14)


# ---------------------------------------------------------------------------
# Panel 3 — Density overlay (contamOption A only)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Panel 4 — Per-point NPL weight scatter (contamOption A only)
# ---------------------------------------------------------------------------
def _draw_weight_scatter(ax, point_est_data: Dict[float, Dict],
                         lams_for_plot: List[float], N: int,
                         contamOption: Optional[str] = None) -> None:
    """
    x-axis : data value
    y-axis : NPL weight assigned to that data point
    One scatter series per lambda (colour-matched to Panel 3 palette).
    Red dashed horizontal line at 1/N (uniform weight reference).
    """
    if not point_est_data:
        ax.text(0.5, 0.5, "No point-est data found",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("NPL Weights vs Data Value", fontsize=13)
        return

    palette = plt.cm.Set2(np.linspace(0, 1, max(len(lams_for_plot), 1)))

    for lam_idx, lam_req in enumerate(lams_for_plot):
        stored_lams = np.array(sorted(point_est_data.keys()))
        closest = float(stored_lams[np.argmin(np.abs(stored_lams - lam_req))])
        rec = point_est_data[closest]

        x_vals   = rec["data_values"]
        weights  = rec["point_weights"]
        sort_idx = np.argsort(x_vals)

        ax.scatter(
            x_vals[sort_idx], weights[sort_idx],
            color=palette[lam_idx], s=8, alpha=0.55,
            zorder=3,
        )

    # Uniform-weight reference line
    ax.axhline(y=1.0 / N, color="crimson", linestyle="--",
               linewidth=1.8, label=rf"$1/N = 1/{N}$", zorder=4)

    ax.set_xlabel("Data value $x_i$", fontsize=16)
    ax.set_ylabel("NPL weight $\\sigma_{w,i}$", fontsize=16)
    ax.set_title(f"NPL Weights vs Data Value (contamOption={contamOption})", fontsize=18)
    ax.legend(fontsize=14, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=14)
    if contamOption == "C":
        ax.set_ylim(0, 2.0 / N)
def _draw_density_overlay(ax, df: pd.DataFrame,
                           theta_true: List[float],
                           lams_for_plot: List[float],
                           eps: float, nu: float, z_outlier: float,
                           contamOption: Optional[str] = None,
                           rho: Optional[float] = None,
                           bootstrap_thetas: Optional[List[List[float]]] = None,
                           wasser_thetas: Optional[List[List[float]]] = None) -> None:
    """
    Blue  : clean Normal(mu_true, sigma_true)
    Red   : (1-eps)*t_nu(mu_true, sigma_true) continuous part
            + dashed vertical line at z_outlier scaled by eps
    Light grey : fitted Normal(mu_hat, sigma_hat) for each bootstrap run
                 in bootstrap_thetas (drawn first so reference curves sit on top)
    Light purple : fitted Normal(mu_hat, sigma_hat) for each Wasserstein bootstrap
    Thin coloured : fitted Normal per lamsForPlot entry from lambda-select df
    """
    mu_true    = float(theta_true[0]) if len(theta_true) > 0 else 0.0
    sigma_true = float(theta_true[1]) if len(theta_true) > 1 else 1.0

    # x-range: cover distribution bulk and relevant outliers
    if contamOption == "A":
        x_lo = min(mu_true - 5 * sigma_true, z_outlier - 1.0)
        x_hi = max(mu_true + 5 * sigma_true, z_outlier + 1.0)
    elif contamOption == "B":
        x_lo = mu_true - 5 * sigma_true
        x_hi = max(mu_true + 5 * sigma_true, 8.0 + 3.0)
    else:
        x_lo = mu_true - 5 * sigma_true
        x_hi = mu_true + 5 * sigma_true

    x = np.linspace(x_lo, x_hi, 600)

    # --- Light background: bootstrap fitted densities (drawn first) ---
    if bootstrap_thetas:
        first_boot = True
        for theta in bootstrap_thetas:
            mu_hat, sigma_hat = theta[0], theta[1]
            if sigma_hat <= 0:
                continue
            # Swapped colors: Bootstrap now light purple
            label = r"B-MRSW bootstrap fitted density" if first_boot else None
            ax.plot(x, norm.pdf(x, loc=mu_hat, scale=sigma_hat),
                    color='#d8b4fe', linewidth=0.8, alpha=0.4,
                    label=label, zorder=1)
            first_boot = False

    # --- Light grey: Wasserstein bootstrap fitted densities ---
    if wasser_thetas:
        first_wasser = True
        for theta in wasser_thetas:
            mu_hat, sigma_hat = theta[0], theta[1]
            if sigma_hat <= 0:
                continue
            # Swapped colors: Wasserstein now grey
            label = r"min $W_2$ bootstrap fitted density" if first_wasser else None
            ax.plot(x, norm.pdf(x, loc=mu_hat, scale=sigma_hat),
                    color='#a0a0a0', linewidth=0.8, alpha=0.35,
                    label=label, zorder=1)
            first_wasser = False

    # --- Blue: clean Normal ---
    ax.plot(x, norm.pdf(x, loc=mu_true, scale=sigma_true),
            color="royalblue", linewidth=2.2, zorder=3,
            label=rf"Clean: $Normal({mu_true},{sigma_true})$")

    # --- Red: contaminated DGP (continuous part) ---
    if contamOption == "A":
        t_pdf = t_dist.pdf(x, df=nu, loc=mu_true, scale=sigma_true)
        ax.plot(x, (1 - eps) * t_pdf,
                color="crimson", linewidth=2.2, zorder=3,
                label=rf"P =  $(0.95)t_{{\nu={int(nu)}}} + (0.05)\delta_{{{int(z_outlier)}}}$")
        # Delta mass at z_outlier: vertical arrow
        ax.annotate(
            "", xy=(z_outlier, eps * 0.85), xytext=(z_outlier, 0),
            arrowprops=dict(arrowstyle="->", color="crimson", lw=2.0)
        )
        ax.text(z_outlier + 0.05, eps * 0.45,
                rf"$\varepsilon\,\delta_{{{z_outlier}}}$",
                color="crimson", fontsize=14)
    elif contamOption == "B":
        # B: (1-eps)*floor(Normal(mu,sigma)/rho)*rho + eps*Normal(8,1)
        # Discrete bulk probabilities
        rho_val = rho if rho is not None else 0.1
        k_min = int(np.floor(x_lo / rho_val))
        k_max = int(np.ceil(x_hi / rho_val))
        ks = np.arange(k_min, k_max + 1)
        xs_discrete = ks * rho_val
        
        probs = norm.cdf(xs_discrete + rho_val, loc=mu_true, scale=sigma_true) - \
                norm.cdf(xs_discrete, loc=mu_true, scale=sigma_true)
        probs *= (1 - eps)
        
        # Plot discrete part as spikes (height = P/rho to match density scale)
        ax.vlines(xs_discrete, 0, probs / rho_val, color="crimson", alpha=0.7, linewidth=1.5, zorder=3)
        
        # Continuous outlier part: Normal(8, 1) * eps
        outlier_pdf = norm.pdf(x, loc=8.0, scale=1.0)
        ax.plot(x, eps * outlier_pdf, color="crimson", linewidth=2.2, zorder=3)
        
        label = rf"P = $({1-eps:.2g}) T_{{{rho_val:.2g}}} \# N({mu_true:.2g}, {sigma_true:.2g}) + ({eps:.2g})N(8,1)$"
        # Proxy for legend
        ax.plot([], [], color="crimson", linewidth=2.2, label=label)
    else:
        # Fallback or C
        pass

    # --- Thin coloured: fitted densities at each requested lambda ---
    if lams_for_plot and not df.empty:
        palette = plt.cm.Set2(np.linspace(0, 1, max(len(lams_for_plot), 1)))
        all_lams = df["lambda"].unique()
        for lam_idx, lam_req in enumerate(lams_for_plot):
            closest = all_lams[np.argmin(np.abs(all_lams - lam_req))]
            subset  = df[np.isclose(df["lambda"], closest)]
            color = palette[lam_idx]
            first = True
            for _, row in subset.iterrows():
                mu_hat    = row.get("theta_0", np.nan)
                sigma_hat = row.get("theta_1", np.nan)
                if np.isnan(mu_hat) or np.isnan(sigma_hat) or sigma_hat <= 0:
                    continue
                label = (rf"Fitted $\lambda={closest:.2g}$" if first else None)
                ax.plot(x, norm.pdf(x, loc=mu_hat, scale=sigma_hat),
                        color=color, linewidth=0.7, alpha=0.5,
                        label=label, zorder=2)
                first = False

    ax.set_xlabel("x", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    ax.set_title(f"Fitted vs Reference Densities (contamOption={contamOption})", fontsize=18)
    leg = ax.legend(fontsize=14, loc="upper right")
    if leg:
        for lh in leg.legend_handles:
            lh.set_alpha(1.0)
            if hasattr(lh, 'set_linewidth'):
                lh.set_linewidth(3.0)

    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=14)


# ---------------------------------------------------------------------------
# Main figure assembly
# ---------------------------------------------------------------------------
def make_figure(df: pd.DataFrame,
                model: str, N: int,
                contamOption: Optional[str],
                theta_true: List[float],
                lams_for_plot: List[float],
                eps: float, nu: float, z_outlier: float,
                rho: Optional[float],
                output_path: str,
                local_bucket: str = "") -> None:

    if df.empty:
        print("[ERROR] No data found — nothing to plot.", file=sys.stderr)
        sys.exit(1)

    df = df.sort_values("lambda").copy()
    df["lambda_str"] = df["lambda"].apply(lambda x: f"{x:.2g}")
    lam_order = _sorted_lam_labels(df)
    df["lambda_str"] = pd.Categorical(
        df["lambda_str"], categories=lam_order, ordered=True)

    # Layout: for normal+{A,B} → 3 panels (W2 boxplot, density, weights); no MSE.
    # All other cases: 2 panels + optional density (contamOption=A) panel.
    normal_contam_diag = (model == "normal" and contamOption in ["A", "B", "C"])

    if normal_contam_diag:
        # 3-panel layout: W2 boxplot | density overlay | weight scatter
        n_panels  = 3
        fig_width = 8 * n_panels
        fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, 5))

        ax_w2   = axes[0]
        ax_dens = axes[1]
        ax_wt   = axes[2]

        contam_label = f", contamOption={contamOption}" if (model == "normal" and contamOption) else ""
        title_w2 = rf"$W_2$ Elbow Plot (model={model}, N={N}{contam_label})"
        draw_w2_boxplot(df, lam_order, ax=ax_w2, title=title_w2)

        # Load all bootstrap fitted densities from the production run
        prod_dir = build_production_path(local_bucket, model, N, contamOption)
        print(f"Reading production bootstrap results from: {prod_dir}")
        bootstrap_thetas = load_bootstrap_thetas(prod_dir)
        print(f"  Loaded {len(bootstrap_thetas)} bootstrap theta estimates.")

        # Load Wasserstein-2 bootstrap results
        wasser_dir = build_wasser_path(local_bucket, model, N, contamOption)
        print(f"Reading Wasserstein results from: {wasser_dir}")
        wasser_thetas = load_bootstrap_thetas(wasser_dir)
        print(f"  Loaded {len(wasser_thetas)} Wasserstein theta estimates.")

        _draw_density_overlay(ax_dens, df, theta_true,
                              [], eps, nu, z_outlier,
                              contamOption=contamOption,
                              rho=rho,
                              bootstrap_thetas=bootstrap_thetas,
                              wasser_thetas=wasser_thetas)

        # Ensure we have at least one lambda to plot weights for
        active_lams = lams_for_plot if lams_for_plot else [2.5]
        
        pt_est_dir = build_point_est_path(local_bucket, model, N, contamOption)
        print(f"Reading point-est results from: {pt_est_dir}")
        point_est_data = load_point_est_data(pt_est_dir, active_lams)
        
        # Update active_lams to the actual snapped values found in the data
        if point_est_data:
            active_lams = sorted(point_est_data.keys())
            
        _draw_weight_scatter(ax_wt, point_est_data, active_lams, N, contamOption=contamOption)

        # --- Custom titles for normal + contamOption=A ---
        lam_val = active_lams[0]
        
        ax_w2.set_title(
            r"$\lambda$ Selection Diagnostic : Elbow Plot of $W_2(F_{\hat{\theta}_{\lambda}}, \hat{Q}_{\lambda}(\hat{\theta}_{\lambda}))$",
            fontsize=18
        )
        
        ax_dens.set_title(
            rf"$\lambda = ({lam_val:.2g})$ B-MRSW vs min Wasserstein-2" + "\n" +
            "(densities of bootstrap sample estimates)",
            fontsize=16
        )
        
        ax_wt.set_title(rf"Data Reweighting for each point in $P_n$ using $\lambda = {lam_val:.2g}$", fontsize=18)
        ax_wt.set_ylabel(r"$\hat{Q}_{\lambda}(\hat{\theta}_{\lambda})$", fontsize=16)

    else:
        # Standard layout: W2 boxplot + MSE [+ density for contamOption=A]
        show_density = (contamOption == "A")
        n_panels     = 2 + int(show_density)
        fig_width    = 8 * n_panels
        fig, axes    = plt.subplots(1, n_panels, figsize=(fig_width, 5))

        ax_w2  = axes[0]
        ax_mse = axes[1]

        contam_label = f", contamOption={contamOption}" if (model == "normal" and contamOption) else ""
        title_w2 = rf"$W_2$ Elbow Plot (model={model}, N={N}{contam_label})"
        draw_w2_boxplot(df, lam_order, ax=ax_w2, title=title_w2)
        
        _draw_mse_curve(ax_mse, df, theta_true, lam_order, model, N, contamOption)

        if show_density:
            ax_dens = axes[2]
            _draw_density_overlay(ax_dens, df, theta_true,
                                  lams_for_plot, eps, nu, z_outlier,
                                  contamOption=contamOption, rho=rho)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved plot to: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Lam-select diagnostic: W2 boxplot + MSE curve "
                    "[+ density overlay for contamOption=A]."
    )
    parser.add_argument("--model", required=True, choices=["gandk", "normal"])
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--contamOption", default=None, choices=["A", "B", "C"])
    parser.add_argument(
        "--theta_true", type=float, nargs="+", default=None,
        help="Ground truth params. Default: (0,1) for normal, (3,1,2,0.5) for gandk.")
    parser.add_argument(
        "--lamsForPlot", type=float, nargs="+", default=[],
        help="Lambda value(s) for which fitted densities are overlaid "
             "(Panel 3, contamOption=A only). E.g. --lamsForPlot 0.01 1.0 100")
    parser.add_argument(
        "--eps", type=float, default=0.05,
        help="Contamination fraction for Panel 3 reference density (default 0.05).")
    parser.add_argument(
        "--nu", type=float, default=12.0,
        help="t-distribution df for Panel 3 reference density (default 12).")
    parser.add_argument(
        "--z", type=float, default=6.0,
        help="Point-mass location for Panel 3 reference density (default 6).")
    parser.add_argument(
        "--rho", type=float, default=0.5,
        help="Discretization step size (contamOption=B only).")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--local_bucket", default=None,
        help="Path to local sbi_sims_local_bucket root. "
             "Defaults to ../../sbi_sims_local_bucket relative to this script.")

    args = parser.parse_args()

    if args.model == "normal" and args.contamOption is None:
        parser.error("--contamOption is required when --model normal")

    if args.theta_true is None:
        args.theta_true = ([0.0, 1.0] if args.model == "normal"
                           else [3.0, 1.0, 2.0, 0.5])

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    local_bucket = args.local_bucket or script_dir

    results_dir = build_results_path(
        local_bucket, args.model, args.N, args.contamOption)
    print(f"Reading results from: {results_dir}")

    if args.output is None:
        contam_tag = f"_contam{args.contamOption}" if args.contamOption else ""
        args.output = os.path.join(
            os.getcwd(),
            f"lam_select_diagnostic_{args.model}_N{args.N}{contam_tag}.pdf")

    df = load_data(results_dir)
    n_lam = df["lambda"].nunique() if not df.empty else 0
    print(f"Loaded {len(df)} records across {n_lam} lambda values.")
    print(f"theta_true = {args.theta_true}")

    make_figure(df, args.model, args.N, args.contamOption,
                args.theta_true, args.lamsForPlot,
                args.eps, args.nu, args.z,
                args.rho,
                args.output,
                local_bucket=local_bucket)


if __name__ == "__main__":
    main()
