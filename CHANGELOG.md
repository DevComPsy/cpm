# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.26.0.dev0] - 2026-09-28

### Added

- Added test units for `cpm.hierarchical.EmpiricalBayes` and `cpm.hierarchical.VariationalBayes`
- Added `number_of_starts` and `initial_guess_supplied` attributes to all optimisers, recording how many starts a fit used and whether the initial guesses were supplied ([#83](https://github.com/DevComPsy/cpm/issues/83))
- Added test units for the new optimiser attributes, and a smoke test that runs a real `cpm.optimisation.Bads` fit
- Added `log_likelihood` and `log_prior` to the output of all optimisers fitted with `prior=True` ([#28](https://github.com/DevComPsy/cpm/issues/28))
- Added a `metrics` argument to all optimisers, for goodness-of-fit metrics such as `PenalisedLikelihoods.BIC` evaluated at the optimum ([#28](https://github.com/DevComPsy/cpm/issues/28))
- Added `cpm.applications.reinforcement_learning.HybridMBMF`, the hybrid model-based / model-free model of the two-step task (Kool et al., 2016; Smid et al., 2022)
- Added `cpm.models.learning.SARSATrace`, a SARSA learning rule with an eligibility trace
- Added test units for `HybridMBMF` and `SARSATrace`
- Added a two-step task example replicating Smid et al. (2022), with its data in `cpm.datasets.load_two_step_data`
- Added `cpm.brainexplorer`, which computes descriptive statistics and applies the exclusion criteria for the BrainExplorer games Space Observer, Scavenger, Treasure Hunt, and Milky Way and Pirate Market, with a page in the API reference
- Added test units for `cpm.brainexplorer`
- Added an example that fits causal ratings in a blocking experiment, recreating Figures 1 and 2 of Spicer et al. (2021), with its data in `cpm.datasets.load_blocking_data`; based on an earlier version by @chotong
- Added `cpm.generators.SessionWrapper`, a `Wrapper` for models that compute all trials of a participant at once
- Added `cpm.models.kernels`, the formulas of the `cpm.models` classes as functions that numba can compile, used by the classes and the built-in applications
- Added numba as an optional dependency (`pip install "cpm-toolbox[numba]"`) that compiles the built-in applications; without it, they give the same results as plain Python
- Added the numba install option to the installation guide, a how-to guide on numba, and troubleshooting entries
- Added a how-to guide on speeding up your own model with `SessionWrapper`, with a test that runs its example
- Added test units for `SessionWrapper`, `cpm.models.kernels`, the fast priors, and the built-in applications with and without numba
- Added a benchmark suite (`benchmarks/run.py` for one evaluation, `benchmarks/hierarchical.py` for the hierarchical tutorials) and `scripts/local_tests.py`, which runs the tests with and without numba
- Added a GitHub Actions workflow that runs the tests with and without numba, on Python 3.11 to 3.14

### Changed

- Rebuilt the documentation with Sphinx and the PyData theme, with a separate API reference, tutorials, examples and how-to guides; pages of the old site redirect
- Converted all docstrings to reStructuredText (NumPy style), and fixed incorrect examples and a `SyntaxWarning` from LaTeX in docstrings
- Restricted package discovery to `cpm`, and added `docs` and `notebooks` optional dependencies
- Sped up long and multi-chain runs of `cpm.hierarchical.EmpiricalBayes` and `cpm.hierarchical.VariationalBayes`
- Made every model cheaper to evaluate: copies of a `cpm.generators.Value` share their prior, and the built-in priors are evaluated without scipy overhead, with identical results
- `cpm.generators.Value.update_prior()` now replaces the prior instead of changing it in place; changing `value.prior.kwds` directly now affects every copy of the `Value`
- `cpm.generators.Wrapper.reset()` now keeps the current priors and bounds of the parameters, instead of restoring the ones the model was created with
- Made the built-in applications (`RLRW`, `HybridMBMF`, `PTSM`, `PTSM1992`, `PTSM2025`) compute all trials at once: 90-220 times faster with numba, 11-23 times without, with the same interface and results
- Made per-trial `cpm.generators.Wrapper` models about twice as fast, by reading trials without `DataFrame.iloc`
- Made the classes in `cpm.models` faster, up to 17 times for large inputs, with identical results
- Made `LogLikelihood.bernoulli` and `LogLikelihood.continuous` 2.5-3 times faster, with identical results
- Made `cpm.applications.signal_detection.EstimatorMetaD` about 10 times faster, with identical results
- `cpm.models.decision.Softmax` no longer overflows: it returns the correct policy where it returned NaN and warned
- `cpm.models.activation.ProspectUtility.weights` and `.utilities` are float arrays instead of object arrays, and the `simulation` records of `HybridMBMF` hold NumPy scalars instead of Python numbers
- `PTSM2025` always uses its power utility: replacing `parameters.utility_curvature` on a model no longer changes it
- Required Python 3.11 or later (previously `>3.11.0`, which excluded 3.11.0), removed the PyPy classifier, and tagged the wheel for Python 3 only

### Fixed

- Fixed the optimisers and `cpm.applications.signal_detection.EstimatorMetaD` raising `UnboundLocalError` for a pandas DataFrame without `ppt_identifier`; they now raise a `ValueError` that explains what to pass, or a `KeyError` that lists the columns if `ppt_identifier` is not one of them, and take `ppt_identifier` from the grouping of data grouped by a single column, such as `data.groupby("ppt")`, so that the fits record the participants
- Fixed `cpm.hierarchical.EmpiricalBayes` and `cpm.hierarchical.VariationalBayes` discarding the estimated population priors after the first evaluation of each fit; hierarchical results change
- Fixed the random starting priors of later chains of `EmpiricalBayes` and `VariationalBayes` being length-one arrays
- Fixed `numpy.random.seed` not reproducing the later chains of `EmpiricalBayes` and `VariationalBayes`, whose starting priors now also respect the lower bound and work with infinite bounds
- Fixed `cpm.optimisation.Bads` emitting a `DeprecationWarning` on every GP fit ([#88](https://github.com/DevComPsy/cpm/issues/88))
- Fixed `cpm.optimisation.Bads` failing under NumPy 2, by requiring `gpyreg>=1.2.1`
- Fixed installs with pandas, matplotlib or pybads releases that do not work with NumPy 2, by requiring `pandas>=2.2.2`, `matplotlib>=3.8.4` and `pybads>=1.0.5`
- Fixed `cpm.optimisation.FminBound` raising `TypeError` on SciPy 1.18.0 and later
- Fixed `cpm.hierarchical.VariationalBayes.ttest` raising `NameError` when `null` is a `pandas.DataFrame`
- Fixed `cpm.hierarchical.VariationalBayes.lmes` repeating the final list of model evidences for every iteration
- Removed a dead, always-zero `mean_errorbar` column from `cpm.hierarchical.VariationalBayes.hyperparameters`
- Fixed `cpm.generators.Value` with `prior="uniform"` spanning `[lower, lower + upper]` instead of `[lower, upper]`
- Fixed `cpm.generators.Parameters.sample()` raising `AttributeError` for a parameter that is `None`
- Fixed callables passed to `cpm.generators.Parameters` leaking onto every other `Parameters` instance
- Fixed `cpm.models.learning.SeparableRule.error` never being filled, which made `noisy_learning_rule()` a no-op, and the `error` of `DeltaRule` and `SeparableRule` having the wrong shape for 1D weights
- Fixed `cpm.generators.Wrapper.reset()` misassigning an array of parameter values when a state is declared before a free parameter
- Fixed `numpy.asarray(value, dtype=...)` raising `TypeError` for a `cpm.generators.Value`
- Fixed `cpm.models.activation.ProspectUtility` with `weighting="prelec"` failing for options with several outcomes
- Fixed `cpm.applications.decision_making.PTSM2025` stopping a fit with `ValueError` when its exponentials overflowed
- Fixed `cpm.models.learning.HumbleTeacher` raising `IndexError` for 1D weights
- Fixed `cpm.generators.Simulator` not raising its intended `TypeError` for an ungrouped `pandas.DataFrame`
- Fixed `cpm.generators.Simulator` raising `AttributeError` for parameters in a `pandas.DataFrame` with grouped data
- Fixed `cpm.generators.Simulator` raising `ValueError` for a single parameter set given as `Parameters`, `dict` or `pandas.Series`
- Fixed `diagnostics()` of `EmpiricalBayes` and `VariationalBayes` failing for three or more free parameters; `convergence_diagnostics_plots` has a new `bounds` argument
- Fixed a clean install raising `ModuleNotFoundError` in `diagnostics()`, by adding `matplotlib` to the dependencies
- Fixed `cpm.utils.data` and `cpm.utils.metad` not being reachable after `import cpm`
- Fixed `cpm.core.diagnostics.gelman_rubin` and `cpm.core.diagnostics.psrf` failing on every call, and `gelman_rubin` computing the statistic incorrectly

## [0.25.6] - 2026-04-15

### Added

- Introduce a third-party connector for the `cpm.generators.Wrapper` class to facilitate integration with external optimisation procedures
- Add validation for 'observed' column in Wrapper class to ensure it exists before running model or computing loss
- Add warnings to inform users if 'observed' column is missing in the data provided to Wrapper class
- Added detailed guidance on fitting the models (PTSM, PTSM1992, and PTSM2025) to data, including recommendations for setting the temperature parameter, handling overflow warnings, and normalizing utility values to avoid numerical instability.
- Added documentations for the datasets included in `cpm.datasets`, detailing the experimental procedure, data structure and usage ([#69](https://github.com/DevComPsy/cpm/pull/69)) @FloorJBurghoorn
- Added `cpm.utils.data.convert_to_RLRW` data conversion utility to create dataframes compatible with `cpm.applications.reinforcement_learning.RLRW`
- Added `cpm.utils.data.convert_to_PTSM` data conversion utility to create dataframes compatible with `cpm.applications.decision_makin.RLRW`

### Changed

- `cpm.core.parallel.execute_parallel` now suppresses ipyparallel cluster status messages by setting log level to ERROR
- `cpm.core.parallel.execute_parallel` `libraries` parameter default is no longer a mutable list
- Update `cpm.applications.reinforcement_learning.RLRW` class to use `numpy.asarray` for the `values` parameter, ensuring compatibility with numpy=>2.0
- Increased bandit task dataset size
- Update test units for `cpm.applications.reinforcement_learning.RLRW` to include handling of new changes, such as using `numpy.asarray` for `values` and adding an 'observed' column in the test dataset
- Improved the error messages in check_nan_and_bounds_in_input to provide more actionable feedback when encountering NaN or Inf values in predicted or observed data, including likely causes and suggested remedies.
- Display option now blocks all prints in optimization module ([#71](https://github.com/DevComPsy/cpm/pull/71)) @tzukpolinsky

### Fixed

- Fix `cpm.core.parallel.in_ipynb` to correctly distinguish Jupyter notebooks from IPython terminal sessions by checking for `ZMQInteractiveShell`
- Fix `cpm.core.parallel.execute_parallel` ipyparallel cluster not being shut down after execution, causing engine process leaks
- Fix wrong probabilities for generating data in `cpm.applications.decision_making.PTSM2025` model [#67](https://github.com/DevComPsy/cpm/issues/67) @FloorJBurghoorn
- Fix matplotlib>=3.10 dependency mismatch errors upon loading `cpm` by removing unused imports in `cpm.utils.metad`
- Fix the usage of ppt_identifier in Minimize class ([#72](https://github.com/DevComPsy/cpm/pull/72)) @tzukpolinsky
- Fix usage of NaN in `cpm.models.decision.SoftMax`
- Fix the import for `cpm.models.learning`

### Removed

- Removed unused imports in `cpm.utils.metad` to prevent dependency issues with matplotlib>=3.10
- Removed `cpm.models.learning.KernelUpdate` due to inconsistencies between equations reported in paper and code available in GitHub

## [0.23.18] - 2025-09-03

### Added

- Add input validation and error handling in all `cpm.optimisation.minimise` methods
- Add test units for `cpm.optimisation.minimise`
- Added three models based on Prospect Theory: `cpm.applications.decision_making.PTSM`, `cpm.applications.decision_making.PTSM1992`, and `cpm.applications.decision_making.PTSM2025` @BenJonathanWagner
- The `cpm.generators.Parameters` class now supports None-type parameters, allowing for more flexible model configurations
- The `cpm.generators.Parameters` class now supports the use of user-defined functions as attributes in addition to freely-varying parameters
- Add `cpm.datasets.load_risky_choices` function to load built-in risky choices dataset @BenJonathanWagner @tuhauser
- Expanded `cpm.models.activation.ProspectUtility` class to include additional parameters for more flexible modeling of decision-making under risk, more closely approximating Tversky & Kahneman's (1992) version of Prospect Theory


### Fixed

- Fix `simulation_export` function to handle DataFrame output correctly
- Fix `detailed_pandas_compiler` function to support new numpy versions
- Fix probability adjustements in `cpm.optimisation.minimise.LogLikelihood` method to ensure correct parameter estimates
- Fix NaN handling in `cpm.models.decision.Softmax`, and `cpm.models.decision.Sigmoid` due to infinities in the exponential function for out-of-bounds parameters

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
