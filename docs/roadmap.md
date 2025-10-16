# Roadmap

We would like to invite anyone interested in contributing to the development of the `cpm` toolbox to join us in shaping its future. This document outlines our current focus areas, ongoing improvements, and planned features.

## Ongoing Improvements

- **Performance**
    - Integrate `numba` for JIT compilation (e.g., with [@jitclass](https://numba.pydata.org/numba-doc/dev/user/jitclass.html))
    - Reduce code duplication and improve organization
    - Optimize `cpm.generators` for faster simulations and data compilation/export
    - Improve shared memory usage and minimize data copies
    - Speed up data cleaning and compilation in modules/exports

- **Documentation**
    - Expand with more examples and tutorials
    - Plan for migration to a dedicated documentation repository

- **Testing**
    - Increase test coverage across all modules

## Planned Features

- **Model Expansion**
    - Add more model components from the literature
    - Provide pre-built, ready-to-deploy models

- **Model Selection**
    - Develop a model selection module with various criteria
    - Ensure compatibility with hierarchical estimation methods

- **Advanced Fitting Methods**
    - Implement new hierarchical and MCMC methods (e.g., Variational Bayes Monte Carlo, Particle filtering)
    - Implement sophisticated and cutting edge evaluation methods (e.g. g-distance, parameter space partitioning, etc.)

## Contributing

We welcome contributions! If you have ideas for new features or improvements, please open an issue to discuss them.
