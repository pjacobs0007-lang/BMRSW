# Complete Experimental Results

This directory contains the complete results of simulations from the paper. 

## Normal Distribution Experiments

EXPERIMENTAL DATA

In folder `normal` there are three subfolders containing the N = 1000 dataset for each of the three contaminated inference settings:

* `normal/experimentalData/contamOption=A/dataset_0.npy`: The N = 1000 observations from the contamination model described in Figure 2 of the main body of the paper.
* `normal/experimentalData/contamOption=B/dataset_0.npy`: The N = 1000 observations from the contamination model described in D.1.2 of the paper.
* `normal/experimentalData/contamOption=C/dataset_0.npy`: The N = 1000 observations from the clean normal distribution used in section D.1.3 of the paper.

B-MRSW RESULTS FILES

* `normal/results/N1000/lamSelectResults/contamOption=A`: Contains all the results of all 225 optimizations performed in the lambda selection diagnostic routine for LEFT panel Figure 2. The results are stored as JSON, contain the value of lambda that was used, and the diagnostic $W2$ value. 

* `normal/results/N1000/contamOption=A`: Contains the results of all 100 optimizations performed in the 100 bootstrap samples for B-MRSW (Figure 2, middle panel).

* `normal/results/N1000/pointEstResults/contamOption=A`: Contains the results of the 1 optimization using B-MRSW, with the fitted weights, which are displayed in figure 2 RIGHT panel.

* `normal/results/N1000/lamSelectResults/contamOption=B`: Contains all the results of all 225 optimizations performed in the lambda selection diagnostic routine for LEFT panel Figure 5. The results are stored as JSON, contain the value of lambda that was used, and the diagnostic $W2$ value. 


* `normal/results/N1000/contamOption=B`: Contains the results of all 100 optimizations performed in the 100 bootstrap samples for B-MRSW (Figure 5, middle panel).

* `normal/results/N1000/pointEstResults/contamOption=B`: Contains the results of the 1 optimization using B-MRSW, with the fitted weights, which are displayed in figure 5 RIGHT panel.

* `normal/results/N1000/lamSelectResults/contamOption=C`: Contains all the results of all 225 optimizations performed in the lambda selection diagnostic routine for LEFT panel Figure 6. The results are stored as JSON, contain the value of lambda that was used, and the diagnostic $W2$ value. 

* `normal/results/N1000/contamOption=C`: Contains the results of all 100 optimizations performed in the 100 bootstrap samples for B-MRSW (Figure 6, middle panel).

* `normal/results/N1000/pointEstResults/contamOption=C`: Contains the results of the 1 optimization using B-MRSW, with the fitted weights, which are displayed in figure 6 RIGHT panel.

MIN-WASSERSTEIN RESULT FILES

* `normal/results/wasser/N1000/contamOption=A` : The results of the 100 bootstrap optimizations for Min Wasserstein (dispalyed in middle panel Figure 2).

* `normal/results/wasser/N1000/contamOption=B` : The results of the 100 bootstrap optimizations for Min Wasserstein (dispalyed in middle panel Figure 5).

* `normal/results/wasser/N1000/contamOption=C` : The results of the 100 bootstrap optimizations for Min Wasserstein (dispalyed in middle panel Figure 6).

## G-and-K Distribution Experiments

EXPERIMENTAL DATA

* `gandk/experimentalData/N1000`: contains all 20 datasets used for the N=1000 coverage experiment with the g-and-k described in Section 5 of the paper.
* `gandk/experimentalData/N5000`: contains all 20 datasets used for the N = 5000 coverage experiment with the g-and-k described in Section 5 of the paper.

B-MRSW RESULT FILES

Note that the JSONs contain a task_id field. The first 100 JSONs in `gandk/results/N1000/lam0.5` map back to results of bootstrap samples for B-MRSW for dataset0 in `gandk/experimentalData/N1000` where the run is with $\lambda = 0.5$, the next 100 to dataset1, and so on. 

So each of the below files contains a total of 2000 optimizations (corresponding to 100 optimiations for each dataset).

* `gandk/results/N1000/lam0.5`
* `gandk/results/N1000/lam1.5`
* `gandk/results/N1000/lam2.5`
* `gandk/results/N1000/lam3.5`

Likewise for the N5000 results, which are stored at:

* `gandk/results/N5000/lam0.5`
* `gandk/results/N5000/lam1.5`
* `gandk/results/N5000/lam2.5`
* `gandk/results/N5000/lam3.5`

NPL-MMD RESULT FILES

The same structure is maintained for the MMD-NPL results, which are stored at:

gandk/results/N1000/npl_mmd/Bandwidth=-1
gandk/results/N1000/npl_mmd/Bandwidth=.15
gandk/results/N1000/npl_mmd/Bandwidth=.3
gandk/results/N1000/npl_mmd/Bandwidth=.5
gandk/results/N1000/npl_mmd/Bandwidth=.65
gandk/results/N1000/npl_mmd/Bandwidth=1

and

gandk/results/N5000/npl_mmd/Bandwidth=-1
gandk/results/N5000/npl_mmd/Bandwidth=.15
gandk/results/N5000/npl_mmd/Bandwidth=.3
gandk/results/N5000/npl_mmd/Bandwidth=.5
gandk/results/N5000/npl_mmd/Bandwidth=.65
gandk/results/N5000/npl_mmd/Bandwidth=1


