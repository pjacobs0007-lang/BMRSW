import argparse
import json
import os
import sys
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(results_dir: str) -> pd.DataFrame:
    """Reads all task JSONs in a directory and returns a DataFrame with lambda and w2 values."""
    if not os.path.exists(results_dir):
        print(f"[WARNING] Results dir not found: {results_dir}", file=sys.stderr)
        return pd.DataFrame()

    rows = []
    for fname in sorted(f for f in os.listdir(results_dir) if f.endswith(".json")):
        try:
            with open(os.path.join(results_dir, fname)) as fh:
                res = json.load(fh)
                
            # Handle different keys that might have been used across iterations
            lam = res.get("lambda") or res.get("lambd")
            w2 = res.get("w2") or res.get("w2_fit_reweighted") or res.get("w_2_fit_reweighted")
            
            if lam is None:
                continue
                
            row = {"lambda": float(lam)}
            if w2 is not None:
                row["w2"] = float(w2)
            
            rows.append(row)
        except Exception as e:
            print(f"[WARNING] Skipping {fname}: {e}", file=sys.stderr)
            
    return pd.DataFrame(rows)


def _sorted_lam_labels(df: pd.DataFrame) -> List[str]:
    """Helper to get a sorted list of lambda string labels."""
    return (df[["lambda", "lambda_str"]].drop_duplicates()
            .sort_values("lambda")["lambda_str"].tolist())


def draw_w2_boxplot(df: pd.DataFrame, lam_order: List[str], ax=None, title: str = None) -> plt.Figure:
    """Creates a standalone or integrated W2 Elbow Plot (boxplot) given a DataFrame of results."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        standalone = True
    else:
        fig = ax.figure
        standalone = False
    
    has_w2 = "w2" in df.columns and df["w2"].notna().any()
    if has_w2:
        df = df.copy()
        df["w2_sqrt"] = np.sqrt(df["w2"])
        sns.boxplot(
            x="lambda_str", y="w2_sqrt",
            data=df, ax=ax,
            hue="lambda_str", palette="viridis",
            order=lam_order, legend=False,
        )
        ax.set_ylabel(
            r"$W_2(F_{\hat{\theta}_{\lambda}}, \hat{Q}_{\lambda}(\hat{\theta}_{\lambda}))$",
            fontsize=16
        )
    else:
        ax.text(0.5, 0.5, "No W2 data found in results.", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="red")

    if title is None:
        title = r"$\lambda$ Selection Diagnostic: $W_2$ Elbow Plot"
        
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(r"$\lambda$", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_xticks(range(len(lam_order)))
    ax.set_xticklabels(lam_order, rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    
    if standalone:
        plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Generate a standalone W2 Elbow Plot (boxplot) for lambda selection from a directory of JSON results."
    )
    parser.add_argument("--results_dir", required=True, 
                        help="Path to directory containing lambda selection JSONs.")
    parser.add_argument("--output_path", default=None, 
                        help="Path to save the PDF. Defaults to w2_boxplot.pdf in the results_dir.")
    parser.add_argument("--title", default=None,
                        help="Optional custom title for the plot.")

    args = parser.parse_args()

    # Determine output path
    if args.output_path is None:
        args.output_path = os.path.join(args.results_dir, "lambda_selection_diagnostic_results.pdf")

    # Load and process data
    print(f"Reading results from: {args.results_dir}")
    df = load_data(args.results_dir)
    
    if df.empty:
        print("[ERROR] No data found in the specified directory. Cannot generate plot.", file=sys.stderr)
        sys.exit(1)

    # Prepare strings for plotting
    df = df.sort_values("lambda").copy()
    df["lambda_str"] = df["lambda"].apply(lambda x: f"{x:.2g}")
    lam_order = _sorted_lam_labels(df)
    df["lambda_str"] = pd.Categorical(
        df["lambda_str"], categories=lam_order, ordered=True
    )

    n_lam = df["lambda"].nunique()
    print(f"Loaded {len(df)} total records across {n_lam} unique lambda values.")

    # Draw and save
    fig = draw_w2_boxplot(df, lam_order, title=args.title)
    fig.savefig(args.output_path, bbox_inches="tight", dpi=150)
    print(f"Saved W2 Boxplot to: {args.output_path}")


if __name__ == "__main__":
    main()
