# CHANGELOG


All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- b921be30: Added Bayesian Adaptive Direct Search (BADS) as an optimization method
- d42e0689: Fmin can incorporate priors into its log likelihood function
- 0a9281f6: Fmin now also returns the hessian matrix of the minimisation function
- 71937516: Parameters can now output parameter bounds if parameter has specified priors
- df4cac2c: FminBound implements a bounded parameter search with L-BFGS-B
- 2fc82638: Parameters now output freely varying parameter names
- abb837ff: Fmin can now reiterate the minimisation process for multiple starting points
- abb837ff: Fmin can now add ppt identifier to the output of the minimisation process
- 2254dbd6: Regenerate initial guesses in Fmin-type optimisation when reset (can be turned off)
- 2254dbd6: EmpiricalBayes creates new starting points for each iteration of the optimisation
- 6780753c: Wrapper updates variables in parameters that are also present in model output
- c8cd4c7c: Simulator.generate() method now expects users to specify what variable to generate
- 7b2571b1: Parameter Recovery now supports the generation of user-specified dependent variables
- 27d16f6b: add squared errors to minimise modules
- 42db58b6: DifferentialEvolution now supports parallelisation

### Changed

- 2477a127: Optimisers now only store freely varying parameter names
- b7ed8069: Refactored Bads to implement up-to-date changes (changed parallelisation, works with new methods in Parameters, implements priors)

### Removed

- 6780753c: Wrapper summary output is removed due to redundancy

### Fixed

- e334d6e8: fix parameter class prior function is not carried over by copy method
- e195266f: fix Wrapper class parameter updates, where list or array inputs deleted Value class attributes of parameters
- 6780753c: Wrapper now correctly finds the number of trials in the model output
- 5f5432bd: -Inf in Loglikelihood is turned into np.finfo(np.float64).min to avoid NaN in the likelihoods
- a84ae319: Parameters now ignores attributes without prior when calculating the PDF
- 32520016: Simulator generated returns an empty array
- 7a276be6: Parameter Recovery quired the wrong dimension to establish what parameters to recover
- cd6ef8cb: Fix naming clashes in parameter recovery
- 57c6a3c0: Fix parallel=False still spawns processes in Optimizations
- 42db58b6: Fixing the issue when likelihood is -inf (bounding to minimum calculable value causes error in sums)
- 42db58b6: Fixing nan and inf checks in the obejctive functions