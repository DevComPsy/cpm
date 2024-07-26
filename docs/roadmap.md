# cpm Roadmap

## Performance improvement

We are always looking for ways to speed up the code. Any contribution aiming to do so is welcome.

Some specific areas we are looking to improve are:

* Using `numba` to speed up the code and allow compilation of classes with [@jitclass](https://numba.pydata.org/numba-doc/dev/user/jitclass.html).
* Reduce code duplication and improve code organization.
* Improve the performance of the `cpm.generators` module to speed up simulations.
* Improve memory usage and reduce the number of copies of the data.
* Improve the data compilation process to speed up the majority of data cleaning happening in the modules.

## Scaling from local to distributed

The big performance-improvement goal is to scale from local to distributed computing.
If you have experience with scaling python libraries, we would really appreciate your help.

## Documentation

We would like to expand the documentation to include more examples and tutorials.
We imagine that these will at some point migrate to their own repository, but for now, they will be part of the main documentation.

## Testing

We would like to improve the testing coverage of the code.

## New features

We are always looking for new features to add to the library.
If you have an idea for a new feature, please open an issue to discuss it.

Two main types of features we planned are:

1. Expansion of the models classes. There are many model components out in the literature and we would like to add as many of them to the library as possible.
2. Add pre-built models to the library as ready-to-deploy applications.
2. Hierarchical and MCMC methods for model fitting.
3. Introducing new classes to scale your simulations from local to distributed computing.
