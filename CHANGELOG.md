# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.23.18] - 2025-09-03

### Added

- Add input validation and error handling in all `cpm.optimisation.minimise` methods
- Add test units for `cpm.optimisation.minimise`
- Added three models based on Prospect Theory: `cpm.applications.decision_making.PTSM`, `cpm.applications.decision_making.PTSM1992`, and `cpm.applications.decision_making.PTSM2025`
- The `cpm.generators.Parameters` class now supports None-type parameters, allowing for more flexible model configurations
- The `cpm.generators.Parameters` class now supports the use of user-defined functions as attributes in addition to freely-varying parameters
- Add `cpm.datasets.load_risky_choices` function to load built-in risky choices dataset
- Expanded `cpm.models.activation.ProspectUtility` class to include additional parameters for more flexible modeling of decision-making under risk, more closely approximating Tversky & Kahneman's (1992) version of Prospect Theory

### Fixed

- Fix `simulation_export` function to handle DataFrame output correctly
- Fix `detailed_pandas_compiler` function to support new numpy versions
- Fix probability adjustements in `cpm.optimisation.minimise.LogLikelihood` method to ensure correct parameter estimates

### Changed

- Resolved a bug in the `detailed_pandas_compiler` function to handle various data types and ensure proper DataFrame formatting in `cpm/core/data.py`.

## [0.23.7] - 2025-07-02

### Added

- Detect parallel method to use given environment (support for parallelisation on Jupyter Notebooks)
- Provide a complete n-dimensional and k-arm reinforcement learning model for multi-armed bandit tasks in applications
- Add support for '>' and '<' operator in Value type
- Add validation to export method and improve data handling in Simulator class
- Update export tests to validate DataFrame output and adjust simulation assertions
- Implement ProspectSoftmaxModel for decision-making under risk
- Add meta-_d_ to applications
- Provide utility functions for data preprocessing with meta-_d_ type models
- Introduce new likelihoods (`multinomial` and `product`)

### Fixed

- Fix multi-outcome log-likelihood calculation in `cpm.optimisation.minimise.LogLikehood.categorical` method
- Fix pandas groupby method for parallelization when in Jupyter Notebook
- Fix magnitude is not taking effect in Nominal
- Fix choice kernel choice should check whether computations still need to carry out
- Fix column assignment logic in simulation_export function
- Fix tests by updating ProspectUtility parameters in tests to reflect changes in constructor
- Fix wrong probability adjustments in `cpm.optimisation.minimise.LogLikelihood` method causing strange parameter estimates
- Fix Issue [#55](https://github.com/DevComPsy/cpm/issues/55): AttributeError: np.float_ was removed in NumPy 2.0 during export

### Changed

- Update model description in RLRW class to include reference to Sutton & Barto (2021)
- Fix column names in `cpm.applications.signal_detection.EstimatordMetaD` class
- Fix `detailed_pandas_compiler` bug to handle various data types and ensure proper DataFrame formatting
- Fix column name issues in `cpm.applications.signal_detection.EstimatordMetaD` class

### Changed

- Improved error handling in `cpm.applications.signal_detection.EstimatordMetaD`
- Improved error handling and added input validation in several methods, such as the `detailed_pandas_compiler` function and parameter bounds handling in `cpm/generators/parameters.py`
- Allow for larger variations in the estimation of the Hessian matrix in test units
- Changed Softmax and Sigmoid function input shape requirements to ensure they accept 1D arrays only, with a warning for 2D arrays

## [Unreleased] <=0.18.4

### Added

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
- b921be30: Added Bayesian Adaptive Direct Search (BADS) as an optimization method
- 42db58b6: DifferentialEvolution now supports parallelisation
- 6312ad99: more thorough computation of inverse Hessian matrix and log determinant of Hessian matrix
- 289cde73: made update_priors usable for both normal and truncated normal priors
- ec2a181c: Implementing Piray's Variational Bayes method
- 2d0c716d: Added convergence diagnostic plots for hierarchical methods

### Changed

- 2477a127: Optimisers now only store freely varying parameter names
- b7ed8069: Refactored Bads to implement up-to-date changes (changed parallelisation, works with new methods in Parameters, implements priors)
- b921393d: rewrote piecewise power function to compute utilities to avoid numpy warnings
- 634b0e87: corrected estimation of parameter variances and means

### Removed

- 6780753c: Wrapper summary output is removed due to redundancy
- f47c684a: remove the redundant pool.join and pool.close

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
- 3e830f64: fix bads value error when unpacking and compiling results from subject-level fits
- ea5b2750: cpm.generators.Simulator can now handle cases where trial numbers differ between participants
- 2ae833f3: cpm.models.learning.DeltaRule.noisy_learning_rule() should not be scaled by learning rate
- 2d0c716d: cpm.hierarchical.EmpiricalBayes non-writable array and np.nanmean reference bug
- 62f92b16: fix #33:optimiser reset fails for parameters with any non-finite bounds
- bfb167a8: updating params in LogParameters should only apply log transform when it is a freely varying parameter
- b2a8ee35: fix LogParameters copy problem
- 5ecada13: fix the issue where updating parameters in LogParameters would only accept non-log values
- 988b77a4: fix variational bayes data type error
- 67df33c3: fix empirical bayes assigning values to objects before creating them
- 62f92b16: fix initial guesses cannot generate starting guesses for parameter with non-finite or nan bounds
- 88d056ff: fix a bug where undeclared variables caused issue in Empirical Bayes
- b094aca9: fix inverted SD in the variational bayes method - remove as it is unnecessary
