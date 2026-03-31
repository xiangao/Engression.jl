module Engression

using Flux
using Statistics
using LinearAlgebra
using Distributions
using ProgressMeter
using Random
using CUDA
using Optimisers

export Engressor, engression, predict, sample

"""
    Engressor

The Engressor model structure.
"""
mutable struct Engressor
    model
    optim_state
    in_dim::Int
    out_dim::Int
    noise_dim::Int
    standardize::Bool
    x_mean
    x_std
    y_mean
    y_std
    device
    beta::Float32
end

"""
    StoLayer(in_dim, out_dim, noise_dim; act=relu)

A stochastic layer that concatenates Gaussian noise to the input.
"""
struct StoLayer
    linear::Dense
    noise_dim::Int
    act
end

Flux.@layer StoLayer

function StoLayer(in_dim::Int, out_dim::Int, noise_dim::Int; act=relu)
    return StoLayer(Dense(in_dim + noise_dim => out_dim), noise_dim, act)
end

function (m::StoLayer)(x::AbstractMatrix)
    noise = eltype(x).(randn(m.noise_dim, size(x, 2)))
    if x isa CuArray
        noise = cu(noise)
    end
    return m.act.(m.linear(vcat(x, noise)))
end

"""
    build_stonet(in_dim, out_dim, hidden_dim, num_layers, noise_dim)

Builds a stochastic neural network.
"""
function build_stonet(in_dim, out_dim, hidden_dim, num_layers, noise_dim)
    layers = []
    # Input layer
    push!(layers, StoLayer(in_dim, hidden_dim, noise_dim, act=relu))
    
    # Hidden layers
    for _ in 1:(num_layers - 2)
        push!(layers, StoLayer(hidden_dim, hidden_dim, noise_dim, act=relu))
    end
    
    # Output layer (linear)
    push!(layers, Dense(hidden_dim => out_dim))
    
    return Chain(layers...)
end

"""
    energy_loss(y_true, y_est1, y_est2, beta=1.0)

Calculates the energy loss: E[||Y - Ŷ||^β] - 0.5 * E[||Ŷ - Ŷ'||^β].
"""
function energy_loss(y_true, y_est1, y_est2, beta=1.0f0)
    # y_true: (out_dim, batch_size)
    # y_est:  (out_dim, batch_size)
    
    # E[||Y - Ŷ||^β]
    diff1 = y_true .- y_est1
    s1 = mean(p -> sum(abs2, p)^(beta/2), eachcol(diff1))
    
    diff2 = y_true .- y_est2
    s1 = (s1 + mean(p -> sum(abs2, p)^(beta/2), eachcol(diff2))) / 2
    
    # E[||Ŷ - Ŷ'||^β]
    diff_est = y_est1 .- y_est2
    s2 = mean(p -> sum(abs2, p)^(beta/2), eachcol(diff_est))
    
    return s1 - 0.5f0 * s2
end

"""
    engression(x, y; kwargs...)

Fits an engression model. x and y should be (obs, dim) matrices or vectors.
"""
function engression(x_raw, y_raw;
                    num_layers=3, hidden_dim=100, noise_dim=10,
                    lr=0.001, num_epochs=1000, batch_size=nothing,
                    standardize=true, beta=1.0, device=nothing)
    
    # Determine device
    if isnothing(device)
        device = CUDA.functional() ? gpu : cpu
    end

    # Format data: Flux expects (dim, obs)
    x = x_raw isa AbstractVector ? reshape(x_raw, 1, :) : collect(x_raw')
    y = y_raw isa AbstractVector ? reshape(y_raw, 1, :) : collect(y_raw')
    
    in_dim, n_obs = size(x)
    out_dim, _ = size(y)
    
    # Standardization
    x_mean, x_std = zeros(Float32, in_dim), ones(Float32, in_dim)
    y_mean, y_std = zeros(Float32, out_dim), ones(Float32, out_dim)
    
    if standardize
        x_mean = mean(x, dims=2)
        x_std = std(x, dims=2)
        x_std[x_std .== 0] .= 1.0f-5
        x = (x .- x_mean) ./ x_std
        
        y_mean = mean(y, dims=2)
        y_std = std(y, dims=2)
        y_std[y_std .== 0] .= 1.0f-5
        y = (y .- y_mean) ./ y_std
    end
    
    # Build model
    model = build_stonet(in_dim, out_dim, hidden_dim, num_layers, noise_dim) |> device
    x = x |> device
    y = y |> device
    
    # Optimizer
    opt = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(opt, model)
    
    # Training loop
    p = Progress(num_epochs, dt=0.5, desc="Training Engression... ")
    
    for epoch in 1:num_epochs
        grads = Flux.gradient(model) do m
            y_hat1 = m(x)
            y_hat2 = m(x)
            energy_loss(y, y_hat1, y_hat2, Float32(beta))
        end
        
        opt_state, model = Optimisers.update(opt_state, model, grads[1])
        
        if epoch % 100 == 0
            loss_val = energy_loss(y, model(x), model(x), Float32(beta))
            next!(p; showvalues=[(:loss, loss_val)])
        else
            next!(p)
        end
    end
    
    return Engressor(model, opt_state, in_dim, out_dim, noise_dim, 
                     standardize, x_mean, x_std, y_mean, y_std, device, Float32(beta))
end

"""
    sample(engressor, x; sample_size=100)

Sample from the fitted conditional distribution.
Returns a tensor of shape (out_dim, obs, sample_size).
"""
function sample(eng::Engressor, x_raw; sample_size=100)
    if x_raw isa AbstractVector
        if length(x_raw) == eng.in_dim && eng.in_dim > 1
            x = reshape(x_raw, :, 1)
        else
            x = reshape(x_raw, 1, :)
        end
    else
        x = collect(x_raw')
    end
    
    if eng.standardize
        x = (x .- eng.x_mean) ./ eng.x_std
    end
    
    x_dev = x |> eng.device
    
    # Generate samples
    # Result shape: (out_dim, obs, sample_size)
    samples = zeros(Float32, eng.out_dim, size(x, 2), sample_size)
    
    for i in 1:sample_size
        y_hat = eng.model(x_dev) |> cpu
        samples[:, :, i] .= y_hat
    end
    
    if eng.standardize
        for i in 1:sample_size
            samples[:, :, i] .= (samples[:, :, i] .* eng.y_std) .+ eng.y_mean
        end
    end
    
    return samples
end

"""
    predict(engressor, x; target="mean", sample_size=100)

Predict quantities from the distribution. target can be "mean", "median", or a quantile (0.1).
Returns an (obs, out_dim) matrix.
"""
function predict(eng::Engressor, x_raw; target="mean", sample_size=500)
    samples = sample(eng, x_raw, sample_size=sample_size)
    
    if target == "mean"
        # mean(samples, dims=3) is (out_dim, obs, 1)
        res = mean(samples, dims=3)
    elseif target == "median"
        res = median(samples, dims=3)
    elseif target isa AbstractFloat
        # eachslice gives out_dim x obs matrices
        res_raw = quantile.(eachslice(samples, dims=(1,2)), target)
        # res_raw is (out_dim, obs)
        res = reshape(res_raw, eng.out_dim, size(samples, 2), 1)
    else
        error("Unknown target type")
    end
    
    # Final result should be (obs, out_dim)
    return collect(dropdims(res, dims=3)')
end

end # module
