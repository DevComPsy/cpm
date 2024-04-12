# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- 6780753c: Wrapper updates variables in parameters that are also present in model output

### Fixed

- 6780753c: Wrapper now correctly finds the number of trials in the model output
- 5f5432bd: -Inf in Loglikelihood is turned into np.finfo(np.float64).min to avoid NaN in the optimization
- a84ae319: Parameters now ignores attributes without prior when calculating the PDF

### Changed

- 084830d2: Simulator input argument is changed from model to wrapper for consistency
- 1e47a565: ProspectUtility outcomes are changed to magnitude

### Removed

- 6780753c: Wrapper summary output is removed due to redundancy