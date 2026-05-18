import argparse
import numpy as np
import os
import sys

# Ensure the parent directory (code/) is in the path so we can import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import sample_normal_groundTruth

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Generate Ground Truth Datasets for Normal Model Study"
    )

    # Ground-truth parameters
    parser.add_argument(
        '--theta_true', type=float, nargs=2, default=[0.0, 1.0],
        metavar=('MU', 'SIGMA'),
        help='True (mu, sigma) of the base Normal distribution (default: 0.0 1.0)'
    )

    # Dataset size and number of replicates
    parser.add_argument('--N', type=int, default=10000,
                        help='Number of observations per dataset (default: 10000)')
    parser.add_argument('--R', type=int, default=20,
                        help='Number of unique datasets to create (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed for reproducibility (default: 42)')

    # Contamination control
    parser.add_argument(
        '--contamOption', type=str, default='C', choices=['A', 'B', 'C'],
        help=(
            "Contamination option:\n"
            "  A: (1-eps)*t_{nu}(mu,sigma) + eps*delta_{z}\n"
            "  B: (1-eps)*floor(Normal(mu,sigma)/rho)*rho + eps*Normal(8,1)\n"
            "  C: Normal(mu, sigma)  [clean, no contamination]"
        )
    )
    parser.add_argument('--eps', type=float, default=0.05,
                        help='Contamination fraction (used by options A and B, default: 0.04)')
    parser.add_argument('--nu', type=float, default=3.0,
                        help='Degrees of freedom for t-distribution (option A only, default: 3.0)')
    parser.add_argument('--z', type=float, default=50.0,
                        help='Point-mass location for delta contamination (option A only, default: 50.0)')
    parser.add_argument('--rho', type=float, default=None,
                        help='Discretization step size (option B only, required for B)')

    args = parser.parse_args()

    # Validate rho for option B
    if args.contamOption == 'B' and (args.rho is None or args.rho <= 0):
        parser.error("--rho must be a positive float when --contamOption=B")

    mu, sigma = args.theta_true

    # Output directory: datasets/{N}/contamOption={A|B|C}
    # We place this relative to the current working directory, 
    # assuming the user runs this from the 'code/' folder.
    output_dir = os.path.join("datasets", "normal", str(args.N), f"contamOption={args.contamOption}")
    os.makedirs(output_dir, exist_ok=True)

    print(
        f"Generating {args.R} datasets | "
        f"theta_true=(mu={mu}, sigma={sigma}) | "
        f"N={args.N} | contamOption={args.contamOption}"
    )
    if args.contamOption == 'A':
        print(f"  eps={args.eps}, nu={args.nu}, z={args.z}")
    elif args.contamOption == 'B':
        print(f"  eps={args.eps}, rho={args.rho}")

    for i in range(args.R):
        curr_seed = args.seed + i
        np.random.seed(curr_seed)

        X_np = sample_normal_groundTruth(
            mu=mu,
            sigma=sigma,
            N=args.N,
            contamOption=args.contamOption,
            eps=args.eps,
            nu=args.nu,
            z=args.z,
            rho=args.rho,
        ).ravel().astype(np.float64)

        fname = os.path.join(output_dir, f"dataset_{i}.npy")
        np.save(fname, X_np)
        print(f"  Saved {fname}")

if __name__ == "__main__":
    main()
