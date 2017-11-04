# Parallel Streaming Wasserstein Barycenters
This repository contains the supporting code for the paper:

[Staib, Matthew and Claici, Sebastian and Solomon, Justin and Jegelka, Stefanie. Parallel Streaming Wasserstein Barycenters. In _Advances in Neural Information Processing Systems 31_, 2017.](https://arxiv.org/abs/1705.07443)

```
@inproceedings{staib2017parallel,
 author = {Staib, Matthew and Claici, Sebastian and Solomon, Justin and Jegelka, Stefanie},
 title = {Parallel Streaming {Wasserstein} Barycenters},
 booktitle = {Advances in Neural Information Processing Systems 31},
 year = {2017}
}
```

## Dependencies
* MPI
* Eigen
* Boost (for RegEx parsing)
* HDF5
* Eigen3-HDF5
* cxxopts
* [MOSEK](https://www.mosek.com/) (optional for estimating Wasserstein distances via LP)

## Getting started
1. First compile the main barycenter function via `make barycenter_mpi`.
2. Then run experiments as in the included shell scripts.
