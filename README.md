# Simulation-Based Inference: B-MRSW and NPL-MMD

This repository contains the code to reproduce the results from the paper.

The code for NPL-MMD comes from the online github repositary associated to the paper \textit{Robust Bayesian Inference for Simulator-based Models via the MMD Posterior Bootstrap} by Dellaporta et al. We are not aware of Licensing information for their repository; we credit them for their code and note it is included here only for reproducibility sake. 

## IMPORTANT NOTE

All methods for which results are reported in the paper use a Bootstrap procedure for uncertainty quantification. For all experiments in the paper, we used Google Cloud Batch to parallelize out the computation over the bootstrap replications. The syntax used to leverage parallel compute infrastucture is very specific to the resource being used. Given that not all users will have access to the same parallel computing resources, we have done the following two things:
- A : We have provided here implementations of all methods that operate SEQUENTIALLY over the bootstrap samples.
    - Note that B-MRSW also uses parallelization within a single optimization (within a round of CMA-ES), and the code is designed to use the available cores on your machine so that a single optimization of B-MRSW is parallelized.
- B : Because using the sequential code is slower, it would take longer to reproduce the experiments that are displayed in the paper. And for this reason, we have also provided access to the final result outputs, which are contained in the folder completeExperimentalResultsOfPaper. In that folder is another README file, which explains how the results are organized and how to find them.

## Setup & Installation

The code uses a collection of commmonly used existing packages (including JAX). To configure your environment, install the dependencies via:

```bash
pip install -r requirements.txt
```

## Repository Structure

- **`datasets/`**: Scripts responsible for generating the clean and contaminated ground-truth datasets (`normalDataCreation.py`, `gandkDataCreation.py`).
- **`npl_mmd/`**: Implementation of the Non-Parametric Learning MMD (NPL-MMD) inference algorithm due to Dellaporta et al. 
- **`experiments/`**: Shell scripts designed to fully automate execution of the experiments.
 --If you want to repeat the generation of Figure 2, you would follow the set of commands in experiments/auxNormalA.txt (which are compacted into a shell script as normalContamOptionA.sh, although it is best to run things one command at a time, because it will give you a better idea of the length of time each step takes given your hardware.
--For Figure 5, see experiments/auxNormalB.txt. 
--For Figure 6, see experiments/auxNormalC.txt
--For Figure 3, see experiments/gandkCoverage.sh | Warning: This is a massive simulation study, and it is better to parallelize this than run it on your machine.
-**`results/`**:If you choose to run the result creation scripts described below, they will stored in the results directory in an organized format so that the visualization scripts can process them.

- **`completeExperimentalResultsOfPaper/`**: Contains the results reported in the paper that came from the parallel compute runs. (see `completeExperimentalResultsOfPaper/README.md` for specific folder documentation).

Additional files:
- **`sga.py`** : Contains the implementation of the SGA algorithm (algorithm 1 of the paper). There is a batch-size argument in its implementation, but it is always set to 1 for experiments.
--**`BMRSW.py`** : Has our implementation of B-MRSW. It contains a function drawSingleBootstrapSample (algorithm 2 of the paper). It has two modes performUQ and lam_select. Lam_select mode executes the procedure described in Section 3.3 of the paper. PerformUQ executes the bootstrap uncertainty quantification for a particular choice of $\lambda$. For additional details of the sytnax, see the code itself (which has documentation), as well as auxNormalA/B/C for example run syntax.
-- **`lambda_selection_diagnostic.py`**: Visualizes the output of BMRSW in lam_select mode. See the CLI for additional details. (One runs lambda select using BMRSW.py first, and then runs this script which points to the directory of results, and then this script generates the visual.)
- **`npl_mmd.py`** : The NPL-MMD implementation (Bayesian Bootstrap on top of Minimum squared MMD with Stochastic Gradient Descent used for optimization.)
- **`bernton_minWasser.py`**: An implemenation of Bernton's minimum Wasserstein distnace approach (Regular boostrap on top of min wasserstein, with Nelder Mead used for optimization)