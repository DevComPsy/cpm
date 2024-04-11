# CHANGELOG


All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- d42e0689: Fmin can incorporate priors into its log likelihood function
- 0a9281f6: Fmin now also returns the hessian matrix of the minimisation function
- 71937516: Parameters can now output parameter bounds

### Changed

### Removed

### Fixed

- e334d6e8: fix parameter class prior function is not carried over by copy method
- e195266f: fix Wrapper class parameter updates, where list or array inputs deleted Value class attributes of parameters