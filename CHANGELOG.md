# CHANGELOG


All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- c8cd4c7: Simulator.generate() method now expects users to specify what variable to generate
- 7b2571b: Parameter Recovery now supports the generation of user-specified dependent variables
- 71a86b8: Added Bayesian Adaptive Direct Search (BADS) as an optimization method

### Changed

### Fixed

- 3252001: Simulator generated returns an empty array
- 7a276be: Parameter Recovery quired the wrong dimension to establish what parameters to recover