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
* HDF5
* [eigen3-hdf5](https://github.com/garrison/eigen3-hdf5)
* [cxxopts](https://github.com/jarro2783/cxxopts)
* [MOSEK](https://www.mosek.com/) (optional for estimating Wasserstein distances via LP)

## Getting started
1. Edit the Makefile to point to your local copys of the dependencies.
1. Compile the main barycenter function via `make barycenter_mpi`.
2. Then run experiments as in the included shell scripts.
