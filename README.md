# # FOMV: Field Operator for Measured Viability

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

**FOMV** is a computational framework to map the viability landscape of a six‑dimensional stochastic dynamical system. It estimates the probability of recovery and the mean first passage time to collapse over a grid of initial conditions for two slow variables (`B` and `M`). The code is parallelized, reproducible, and includes interactive 3D visualizations.

## Features

- Efficient noise generation using Beta(2,2) distribution.
- Markov chain sampling of fast variables (`E`, `G`, `T`, `C`).
- Massive parallel simulation with `multiprocessing`.
- Bootstrap confidence intervals for MFPT.
- Interactive Plotly widgets for exploratory data analysis.

## Requirements

- Python 3.9+
- See `requirements.txt` for dependencies.

## Quick Start

### Local installation

```bash
git clone https://github.com/omorales/fomv.git
cd fomv
pip install -r requirements.txt
python fomv/core.py
