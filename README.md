# Engression.jl

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`Engression.jl` is a Julia implementation of engression, a distributional
regression method based on the energy score. Instead of estimating only
`E[Y | X]`, it fits a stochastic neural network that can be sampled from
`P(Y | X)`.

I use this package in the distributional DiD and mediation examples, where the
counterfactual object is a distribution rather than a single conditional mean.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/xiangao/Engression.jl")
```

## Quickstart

### Basic Usage

```julia
using Engression
using Random

# Generate synthetic data
Random.seed!(123)
n = 1000
x = randn(n, 1)
# Heteroskedastic noise: y = x + |x| * noise
y = x .+ abs.(x) .* randn(n, 1)

# Fit engression model
model = engression(x, y; num_epochs=1000, hidden_dim=64)

# Predict conditional mean
y_mean = predict(model, x, target="mean")

# Predict conditional median
y_median = predict(model, x, target="median")

# Predict 90th percentile
y_q90 = predict(model, x, target=0.9)

# Sample from the conditional distribution
# Returns a tensor of shape (out_dim, obs, sample_size)
samples = sample(model, x; sample_size=100)
```

## What is included

- Energy-score training for stochastic neural networks.
- Conditional means, medians, quantiles, and simulated draws.
- CUDA support through Flux and CUDA when a GPU is available.

## API Reference

### `engression(x, y; kwargs...)`
Fits the model. `x` and `y` can be matrices (observations in rows) or vectors.
- `num_layers`: Number of layers in the network (default: 3).
- `hidden_dim`: Hidden layer dimension (default: 100).
- `noise_dim`: Dimension of the noise injected (default: 10).
- `lr`: Learning rate (default: 0.001).
- `num_epochs`: Training epochs (default: 1000).
- `standardize`: Whether to standardize inputs (default: true).

### `predict(model, x; target="mean", sample_size=500)`
Predict quantities from the distribution.
- `target`: "mean", "median", or a float (e.g., 0.1 for 10th percentile).
- `sample_size`: Number of samples used to approximate the quantity.

### `sample(model, x; sample_size=100)`
Generate samples from the conditional distribution $P(Y|X)$.
- Returns a tensor of shape `(out_dim, obs, sample_size)`.

## License
MIT
