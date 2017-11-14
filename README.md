# Parallel Streaming Wasserstein Barycenters
This repository contains the supporting code for the paper:

[Matthew Staib, Sebastian Claici, Justin Solomon, Stefanie Jegelka. Parallel Streaming Wasserstein Barycenters. In _Advances in Neural Information Processing Systems 31_, 2017.](https://arxiv.org/abs/1705.07443)

```
@inproceedings{staib2017parallel,
 author = {Staib, Matthew and Claici, Sebastian and Solomon, Justin and Jegelka, Stefanie},
 title = {Parallel Streaming {Wasserstein} Barycenters},
 booktitle = {Advances in Neural Information Processing Systems 31},
 year = {2017}
}
```

## Dependencies
### Software
* MPI
* [Eigen](http://eigen.tuxfamily.org/)
* [HDF5](https://support.hdfgroup.org/HDF5/)
* [eigen3-hdf5](https://github.com/garrison/eigen3-hdf5)
* [cxxopts](https://github.com/jarro2783/cxxopts)
* [MOSEK](https://www.mosek.com/) (optional for estimating Wasserstein distances via LP)

### Data
* [UCI skin segmentation dataset](https://archive.ics.uci.edu/ml/datasets/skin+segmentation). We slightly changed the format and hence include a local copy in `input_data`.

## Getting started
1. Edit the Makefile to point to your local copys of the dependencies.
2. Compile the main barycenter function via `make barycenter_mpi`.
3. Then experiments are run by calling barycenter_mpi with various arguments. We include example shell scripts `run_vmf_experiment.sh` and `run_skin_experiment_full.sh`. Full explanation of the various parameters can be found below (and also in `parse_args.cpp`:
```i,iters --  Number of iterations to run each thread
e,experiment --  Which experiment to run (skin,vmf,logit,gaussian) (the first two were run for the paper)
s,subsets --  Number of subsets to split into (for WASP)
k,skip --  Number of timesteps between MCMC samples
N,support --  Number of support points
o,outdir --  Output directory for .h5 files
d,saveincrement --  How often to save .h5 files
a,stepsize --  Stepsize for gradient ascent
w,movingwindow --  Width of histogram moving window (or 0 to keep full history)
m,driftrate --  Rate of drift of VMF distributions
b,burniniters --  Number of burn-in iters for MCMC chain
f,fullsampler --  Whether to get samples from the full MCMC chain
p,datapoints --  Number of datapoints to use (for the skin example; useful for testing more quickly)
```
4. Plots can then be generated via the included Matlab scripts (be sure to point these scripts to your output directory!). Specifically,
  * `skin_wasp_compare` compares our stochastic approach to the standard linear programming barycenter algorithm for WASP.
  * `plot_cpp_output_vmf` produces output images for our Von Mises-Fisher experiments
  * `plot_cpp_output_skin` gives some example diagnostic plots for the UCI experiments
  * the scripts in `scripts` produce convergence plots of our barycenter estimate for the UCI experiments
