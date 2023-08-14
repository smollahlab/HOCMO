
HOCMO
=====

This repository implements the non-negative tensor factorization on multi-omics data to deconvolute the epigenetic microenvironment in breast cancer.

## Data

Two datasets were used to model the high-order correlations between signaling proteins, growth ligands and histone modifications:

- RPPA (reverse phase protein array): We downloaded the level3 (log 2 normalized) RPPA data from the [Synapse platform](https://www.synapse.org/#!Synapse:syn12555331).
- GCP (global chromatin profiles): We downloaded the level 3 (log2 normalized) GCP data from the [Synapse platform](https://www.synapse.org/#!Synapse:syn18491838).

## Requirement

  * Python 3.6
  * Numpy
  * Pandas

## Usage

1. The fatorization.py implements non-negative tensor factorization on RPPA and GCP tensors using NCP.
2. The main.py runs the pipeline end-to-end for RPPA and GCP datasets. 

## Reference

[1] Kolda, T.G. and Bader, B.W., 2009. Tensor decompositions and applications. SIAM review, 51(3), pp.455-500.

[2] Kim, J., He, Y. and Park, H., 2014. Algorithms for nonnegative matrix and tensor factorizations: A unified view based on block coordinate descent framework. Journal of Global Optimization, 58(2), pp.285-319.

[3] Kossaifi, J., Panagakis, Y., Anandkumar, A. and Pantic, M., 2016. Tensorly: Tensor learning in python. arXiv preprint arXiv:1610.09555.

## Citation

Min Shi, Rintsen Sherpa, Liubou Klindziuk, Stefanie Kriel, Shamim Mollah. “HOCMO: A higher-order correlation model to deconvolute epigenetic microenvironment in breast cancer”. 2020. doi: https://www.biorxiv.org/content/10.1101/2020.12.01.406249v2

