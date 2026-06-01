using Engression
using Test
using Statistics
using Random
using Flux

@testset "Engression.jl" begin
    Random.seed!(123)
    
    # Synthetic Data: Y = X^2 + noise
    n = 200
    x = randn(Float32, n, 1)
    y = x.^2 .+ 0.1f0 .* randn(Float32, n, 1)
    
    # Fit model (few epochs for speed)
    model = engression(x, y, num_epochs=200, hidden_dim=50, noise_dim=5)
    
    @test model.in_dim == 1
    @test model.out_dim == 1
    
    # Predict
    y_pred = predict(model, x, target="mean")
    @test size(y_pred) == (n, 1)
    
    # Check correlation (should be reasonable even with 200 epochs)
    correlation = cor(vec(y), vec(y_pred))
    @test correlation > 0.5
    
    # Sample
    samples = sample(model, x[1:2, :], sample_size=10)
    @test size(samples) == (1, 2, 10)
    
    # Quantile
    q50 = predict(model, x[1:5, :], target=0.5)
    @test size(q50) == (5, 1)
end

@testset "energy_loss: finite gradient at exact matches (NaN-gradient regression)" begin
    # Regression test for a NaN bug: the Euclidean norm ‖v‖ = (Σv²)^(β/2) has an
    # infinite gradient at v = 0. With a scalar outcome (out_dim = 1) the network easily
    # produces an exact match (prediction == target, or the two noise draws coinciding),
    # which used to yield a NaN gradient and poison training (loss -> NaN). The ε-smoothed
    # loss must keep both value and gradient finite.
    y  = reshape(Float32[1.0, 2.0, 3.0, 4.0], 1, :)
    y2 = y .+ 0.3f0

    # value is finite even when y_est1 exactly equals y_true
    @test isfinite(Engression.energy_loss(y, y, y2))

    # gradient w.r.t. a prediction that exactly matches the target is finite (the bug)
    g1 = Flux.gradient(ŷ -> Engression.energy_loss(y, ŷ, y2), y)[1]
    @test all(isfinite, g1)

    # gradient is also finite when the two estimates coincide (Ŷ - Ŷ' = 0)
    g2 = Flux.gradient(ŷ -> Engression.energy_loss(y2, ŷ, ŷ), y)[1]
    @test all(isfinite, g2)
end
