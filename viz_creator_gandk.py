import argparse
import json
import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.stats import norm

from lambda_selection_diagnostic_viz import draw_w2_boxplot, load_data, _sorted_lam_labels

def process_results_from_data(method_name, data_list, theta_true):
    """Processes a list of result data objects."""
    if not data_list:
        return None

    # groups[dataset_id] = list of theta_est (each is length 4)
    groups = {} 
    for data in data_list:
        try:
            d_id = data.get('dataset_id', '0')
            theta_est = np.array(data['theta_est'])
            if d_id not in groups:
                groups[d_id] = []
            groups[d_id].append(theta_est)
        except Exception:
            continue

    dataset_ids = sorted(groups.keys())
    if not dataset_ids:
        return None

    median_estimator_mses = {0: [], 1: [], 2: [], 3: []} # One MSE list per parameter
    coverage_counts = np.zeros(4) # [a, b, g, k]
    all_widths = [[] for _ in range(4)] # To compute median

    for d_id in dataset_ids:
        estimates = np.array(groups[d_id]) # Shape (M, 4)
        
        # 1. Posterior Median Estimator for this dataset
        theta_median = np.median(estimates, axis=0) # Shape (4,)
        
        # 2. Coverage and Width for each parameter (95% CI), and MSE of median
        for p in range(4):
            # MSE of median
            mse = (theta_median[p] - theta_true[p])**2
            median_estimator_mses[p].append(mse)
            
            # Coverage
            ci_low = np.percentile(estimates[:, p], 2.5)
            ci_high = np.percentile(estimates[:, p], 97.5)
            width = ci_high - ci_low
            all_widths[p].append(width)
            if ci_low <= theta_true[p] <= ci_high:
                coverage_counts[p] += 1

    num_datasets = len(dataset_ids)
    coverage_rates = coverage_counts / num_datasets
    median_widths = [np.median(all_widths[p]) for p in range(4)]
    
    # Final aggregation across datasets
    param_mses = [np.mean(median_estimator_mses[p]) for p in range(4)]

    return {
        "method": method_name,
        "coverage": coverage_rates,
        "median_widths": median_widths,
        "param_mses": param_mses
    }

def collect_ot_results(base_dir, theta_true):
    """Collects and groups OT results by sample size and lambda.
    Path format: results/gandk/bmrsw/N{N}/performUQ/lam_{lam}/dataset_{i}/*.json
    """
    all_ot_results = []
    
    for n_dir in ["N1000", "N5000"]:
        n_path = os.path.join(base_dir, "bmrsw", n_dir, "performUQ")
        if not os.path.exists(n_path):
            continue
            
        for lam_dir in os.listdir(n_path):
            if not lam_dir.startswith("lam_"):
                continue
            lam_val = lam_dir.replace("lam_", "")
            lam_path = os.path.join(n_path, lam_dir)
            
            data_list = []
            for dataset_dir in os.listdir(lam_path):
                if not dataset_dir.startswith("dataset_"):
                    continue
                d_id = dataset_dir.replace("dataset_", "")
                ds_path = os.path.join(lam_path, dataset_dir)
                
                for fname in os.listdir(ds_path):
                    if fname.endswith(".json"):
                        with open(os.path.join(ds_path, fname), 'r') as f:
                            try:
                                data = json.load(f)
                                data["dataset_id"] = d_id
                                data_list.append(data)
                            except:
                                pass
            
            if data_list:
                method_name = f"Robust OT-SBI ($\\lambda={lam_val}$)"
                res = process_results_from_data(method_name, data_list, theta_true)
                if res:
                    res['N'] = int(n_dir.replace("N", ""))
                    res['lambda'] = str(lam_val)
                    res['bandwidth'] = '--'
                    res['method_short'] = "B-MRSW"
                    all_ot_results.append(res)
                    
    return all_ot_results

def collect_npl_results(base_dir, theta_true):
    """Collects and groups NPL results by sample size and bandwidth.
    Path format: results/gandk/npl_mmd/N{N}/bandwidth_{bw}/dataset_{i}/*.json
    """
    all_npl_results = []
    
    for n_dir in ["N1000", "N5000"]:
        n_path = os.path.join(base_dir, "npl_mmd", n_dir)
        if not os.path.exists(n_path):
            continue
            
        for bw_dir in os.listdir(n_path):
            if not bw_dir.startswith("bandwidth_"):
                continue
            bw_val = bw_dir.replace("bandwidth_", "")
            bw_path = os.path.join(n_path, bw_dir)
            
            data_list = []
            for dataset_dir in os.listdir(bw_path):
                if not dataset_dir.startswith("dataset_"):
                    continue
                d_id = dataset_dir.replace("dataset_", "")
                ds_path = os.path.join(bw_path, dataset_dir)
                
                for fname in os.listdir(ds_path):
                    if fname.endswith(".json"):
                        with open(os.path.join(ds_path, fname), 'r') as f:
                            try:
                                data = json.load(f)
                                data["dataset_id"] = d_id
                                data_list.append(data)
                            except:
                                pass
                                
            if data_list:
                method_name = f"NPL-MMD (bandwidth = {bw_val})"
                res = process_results_from_data(method_name, data_list, theta_true)
                if res:
                    res['N'] = int(n_dir.replace("N", ""))
                    res['lambda'] = '--'
                    try:
                        if float(bw_val) == -1.0:
                            res['bandwidth'] = 'Med. Heuristic'
                        else:
                            res['bandwidth'] = str(bw_val)
                    except ValueError:
                        res['bandwidth'] = str(bw_val)
                    res['method_short'] = "NPL-MMD"
                    all_npl_results.append(res)
                    
    return all_npl_results

def generate_latex_table(results, output_path):
    """Generates a LaTeX table from the results."""
    with open(output_path, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\resizebox{\\textwidth}{!}{\n")
        f.write("\\begin{tabular}{llcc | cc | cc | cc | cc}\n")
        f.write("\\hline\n")
        f.write("N & Method & $\\lambda$ & Bandwidth & \\multicolumn{2}{c|}{$a$} & \\multicolumn{2}{c|}{$b$} & \\multicolumn{2}{c|}{$g$} & \\multicolumn{2}{c}{$k$} \\\\\n")
        f.write(" & & & & Cov/Wid & MSE & Cov/Wid & MSE & Cov/Wid & MSE & Cov/Wid & MSE \\\\\n")
        f.write("\\hline\n")
        
        last_n = None
        for res in results:
            if res is None: continue
            
            method_short = res.get('method_short', res['method'])
            lam = res.get('lambda', '--')
            bw = res.get('bandwidth', '--')
            
            if "NPL-MMD" in method_short:
                try:
                    if float(bw) in [0.65, 1.0]:
                        continue
                except ValueError:
                    pass
            
            if last_n is not None and res['N'] != last_n:
                f.write("\\noalign{\\hrule height 1.5pt}\n")
            last_n = res['N']
            
            cov = res['coverage']
            wid = res['median_widths']
            mses = res['param_mses']
            
            # Format each parameter cell as "Cov (Wid) & MSE"
            p_cells = [f"{cov[i]:.2f} ({wid[i]:.2f}) & {mses[i]:.4f}" for i in range(4)]
            f.write(f"{res['N']} & {method_short} & {lam} & {bw} & {' & '.join(p_cells)} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("}\n")
        f.write("\\caption{Comparison of G-and-K Inference Methods: Coverage, Median Interval Width (Wid), and MSE of Posterior Median per parameter}\n")
        f.write("\\label{tab:gandk_comparison}\n")
        f.write("\\end{table}\n")
    print(f"LaTeX table saved to {output_path}")

def gandk_quantile(p, theta, c=0.8):
    """Calculates the quantile function of the G-and-K distribution."""
    a, b, g, k = theta
    z = norm.ppf(p)
    term1 = 1 + c * np.tanh(g * z / 2.0)
    term2 = (1 + z**2)**k
    return a + b * term1 * term2 * z

def plot_quantile_comparison(theta_true, eps=0.05, rho=0.05, z_outlier=50, ax=None):
    """Creates a plot comparing the true G-and-K quantile function with the contaminated/rounded one."""
    ps = np.linspace(0.01, 0.99, 2000)
    q_true = gandk_quantile(ps, theta_true)
    
    # Calculate mixture threshold: p_lower = (1-eps) * F_GK(z_outlier)
    p_lower = 1 - eps
    p_upper = p_lower + eps
    
    q_mix = np.zeros_like(ps)
    for i, p in enumerate(ps):
        if p < p_lower:
            q_mix[i] = gandk_quantile(p / (1 - eps), theta_true)
        elif p <= p_upper:
            q_mix[i] = z_outlier
        else:
            q_mix[i] = gandk_quantile((p - eps) / (1 - eps), theta_true)
            
    q_contam = np.floor(q_mix / rho) * rho
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(ps, q_true, 'b-', label=f'Clean: G-and-K({theta_true[0]},{theta_true[1]},{theta_true[2]},{theta_true[3]})')
    ax.plot(ps, q_contam, 'r-', label='Contaminated Distribution')
    ax.axvline(x=1-eps, color='green', linestyle='-.', label=f'1-eps ({1-eps:.2f})')
    ax.set_ylim(bottom=1.5, top=10)
    ax.set_xlabel('Quantile (p)', fontsize=20)
    ax.set_ylabel('Q(p)', fontsize=20)
    ax.set_title(f'G-and-K Experiment: Quantile Functions', fontsize=22)
    ax.legend(loc='upper left', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True, alpha=0.3)


def plot_sensitivity_panels(ax_left, ax_right, all_results, metric_key='param_mses'):
    """Plots sensitivity panels (MSE or Coverage) for both methods."""
    ot_data = {'1000': [], '5000': []}
    npl_data = {'1000': [], '5000': []}
    
    for res in all_results:
        N = str(res['N'])
        method = res['method']
        vals = res[metric_key]
        
        if "Robust OT-SBI" in method and "SelectResults" not in method:
            try:
                lam = float(method.split('=')[1].replace(')', ''))
                ot_data[N].append((lam, vals))
            except Exception: pass
            
        elif "NPL-MMD" in method:
            try:
                bw_str = method.split('=')[1].replace(')', '').strip()
                bw = float(bw_str)
                if bw != -1.0:
                    npl_data[N].append((bw, vals))
            except Exception: pass
            
    for N in ['1000', '5000']:
        ot_data[N].sort(key=lambda x: x[0])
        npl_data[N].sort(key=lambda x: x[0])
        
    colors = ['C0', 'C1', 'C2', 'C3']
    labels = ['a', 'b', 'g', 'k']
    
    # Left Panel (OT Lambda)
    for N, linestyle in [('1000', '--'), ('5000', '-')]:
        if not ot_data[N]: continue
        lams = [x[0] for x in ot_data[N]]
        y_vals = np.array([x[1] for x in ot_data[N]])
        for i in range(4):
            label = labels[i] if N == '5000' else "_nolegend_"
            ax_left.plot(lams, y_vals[:, i], color=colors[i], linestyle=linestyle, marker='o', label=label, linewidth=2)
            
    if metric_key == 'param_mses':
        ax_left.set_yscale('log')
        ax_left.set_ylabel("Avg. MSE of Median Estimator", fontsize=20)
        ax_left.set_title(r"B-MRSW: MSE vs $\lambda$", fontsize=22)
    else:
        ax_left.set_ylabel("Empirical Coverage Rate", fontsize=20)
        ax_left.set_title(r"B-MRSW: Coverage vs $\lambda$", fontsize=22)
        ax_left.set_ylim(0, 1.05)
        ax_left.axhline(0.95, color='gray', linestyle=':', alpha=0.8)
        ax_left.set_yticks([0, 0.25, 0.5, 0.75, 0.95, 1.0])
        
    ax_left.set_xlabel(r"$\lambda$", fontsize=20)
    ax_left.tick_params(axis='both', which='major', labelsize=16)
    ax_left.grid(True, alpha=0.3)
    
    # Right Panel (NPL Bandwidth)
    for N, linestyle in [('1000', '--'), ('5000', '-')]:
        if not npl_data[N]: continue
        bws = [x[0] for x in npl_data[N]]
        y_vals = np.array([x[1] for x in npl_data[N]])
        for i in range(4):
            label = labels[i] if N == '5000' else "_nolegend_"
            ax_right.plot(bws, y_vals[:, i], color=colors[i], linestyle=linestyle, marker='o', label=label, linewidth=2)
            
    ax_right.set_xlabel("Bandwidth", fontsize=20)
    if metric_key == 'param_mses':
        ax_right.set_title("NPL-MMD: MSE vs Bandwidth", fontsize=22)
    else:
        ax_right.set_title("NPL-MMD: Coverage vs Bandwidth", fontsize=22)
        ax_right.set_ylim(0, 1.05)
        ax_right.axhline(0.95, color='gray', linestyle=':', alpha=0.8)
        ax_right.set_yticks([0, 0.25, 0.5, 0.75, 0.95, 1.0])
        
    ax_right.tick_params(axis='both', which='major', labelsize=16)
    ax_right.grid(True, alpha=0.3)


def generate_diagnostic_panel_part3(all_results, theta_true, base_dir, output_path="gandk_diagnostic_panel_part3.pdf"):
    """Creates a 2-row panel:
    Top row (2 panels): Quantile (A), Lambda Select (B)
    Bottom row (4 panels): OT MSE (C), NPL MSE (D), OT Cov (E), NPL Cov (F)
    """
    fig = plt.figure(figsize=(24, 12))
    gs = fig.add_gridspec(2, 4, hspace=0.4, wspace=0.3)
    
    # --- Row 1 ---
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    
    # --- Row 2 ---
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[1, 2])
    ax_f = fig.add_subplot(gs[1, 3])
    
    # Share Y-axes for paired sensitivity plots
    ax_d.sharey(ax_c)
    ax_f.sharey(ax_e)
    
    # Panel A: Quantile Comparison
    plot_quantile_comparison(theta_true, ax=ax_a)
    rect_a = Rectangle((0, 0), 1, 1, transform=ax_a.transAxes, linewidth=3, edgecolor='black', facecolor='none', clip_on=False)
    ax_a.add_patch(rect_a)
    ax_a.text(0.98, 0.98, 'A', transform=ax_a.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
    
    # Panel B: Lambda Selection
    lam_select_dir = os.path.join(base_dir, "bmrsw", "N5000", "lamSelect")
    if os.path.exists(lam_select_dir):
        df_lam = load_data(lam_select_dir)
        if not df_lam.empty:
            df_lam = df_lam.sort_values("lambda").copy()
            df_lam["lambda_str"] = df_lam["lambda"].apply(lambda x: f"{x:.2g}")
            lam_order = _sorted_lam_labels(df_lam)
            df_lam["lambda_str"] = pd.Categorical(df_lam["lambda_str"], categories=lam_order, ordered=True)
            
            title_b = r"$\lambda$ Selection Diagnostic : Elbow Plot of $W_2(F_{\hat{\theta}_{\lambda}}, \hat{Q}_{\lambda}(\hat{\theta}_{\lambda}))$ (N=5000)"
            draw_w2_boxplot(df_lam, lam_order, ax=ax_b, title=title_b)
        else:
            ax_b.text(0.5, 0.5, "No Lambda Selection Data", ha='center', va='center', fontsize=20)
    else:
        ax_b.text(0.5, 0.5, "Lambda Selection Dir Not Found", ha='center', va='center', fontsize=20)

    rect_b = Rectangle((0, 0), 1, 1, transform=ax_b.transAxes, linewidth=3, edgecolor='black', facecolor='none', clip_on=False)
    ax_b.add_patch(rect_b)
    ax_b.text(0.98, 0.98, 'B', transform=ax_b.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
    
    # Panel C & D: MSE Sensitivity
    plot_sensitivity_panels(ax_c, ax_d, all_results, metric_key='param_mses')
    for ax, lbl in [(ax_c, 'C'), (ax_d, 'D')]:
        rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes, linewidth=3, edgecolor='black', facecolor='none', clip_on=False)
        ax.add_patch(rect)
        ax.text(0.98, 0.98, lbl, transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
        
    # Panel E & F: Coverage Sensitivity
    plot_sensitivity_panels(ax_e, ax_f, all_results, metric_key='coverage')
    for ax, lbl in [(ax_e, 'E'), (ax_f, 'F')]:
        rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes, linewidth=3, edgecolor='black', facecolor='none', clip_on=False)
        ax.add_patch(rect)
        ax.text(0.98, 0.98, lbl, transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
        
    # Add shared legend
    import matplotlib.lines as mlines
    line_1000 = mlines.Line2D([], [], color='black', linestyle='--', label='N = 1000')
    line_5000 = mlines.Line2D([], [], color='black', linestyle='-', label='N = 5000')
    
    handles_f, labels_f = ax_f.get_legend_handles_labels()
    if handles_f:
        ax_f.legend(handles=handles_f + [line_1000, line_5000], fontsize=12, loc='lower right', ncol=2, frameon=True, framealpha=0.8)
    
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Diagnostic Panel Part 3 saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create 6-Panel G-and-K Diagnostic Viz (Part 3)")
    parser.add_argument("--theta_true", type=float, nargs=4, default=[3.0, 1.0, 2.0, 0.5])
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--z_outlier", type=float, default=50.0)
    parser.add_argument("--base_dir", type=str, default="results/gandk", help="Path to results/gandk directory")
    parser.add_argument("--output_path", type=str, default="gandk_diagnostic_panel_part3.pdf")
    parser.add_argument("--output_table", type=str, default="gandk_comparison_table.tex", help="Path to save LaTeX table")
    
    args = parser.parse_args()
    
    print("Collecting B-MRSW results...")
    ot_res = collect_ot_results(args.base_dir, args.theta_true)
    print(f"Found {len(ot_res)} B-MRSW evaluation configurations.")
    
    print("Collecting NPL-MMD results...")
    npl_res = collect_npl_results(args.base_dir, args.theta_true)
    print(f"Found {len(npl_res)} NPL-MMD evaluation configurations.")
    
    all_results = ot_res + npl_res
    
    if not all_results:
        print("[WARNING] No performance results found. The sensitivity plots will be empty.")
    else:
        generate_latex_table(all_results, args.output_table)
        
    generate_diagnostic_panel_part3(all_results, args.theta_true, args.base_dir, args.output_path)

if __name__ == "__main__":
    main()
