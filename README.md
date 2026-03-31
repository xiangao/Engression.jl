# Engression.jl

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`Engression.jl` is a Julia implementation of **Engression**, a distributional regression method based on Energy Loss. It allows you to estimate the entire conditional distribution $P(Y|X)$ by training a stochastic neural network.

This package is a Julia port of the `engression` algorithm, which is particularly useful for tasks where the conditional distribution is non-Gaussian, multi-modal, or heteroskedastic.

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

## Key Features

- **Energy Loss:** Uses a robust loss function that doesn't require parametric assumptions about the error distribution.
- **Stochastic Neural Network:** The model captures the distribution by injecting noise into the hidden layers.
- **GPU Support:** Automatically uses CUDA if available (via Flux.jl and CUDA.jl).
- **Flexible Predictions:** Easily extract means, medians, quantiles, or full samples from the estimated distribution.

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
