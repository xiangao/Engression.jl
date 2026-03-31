using Engression
using Test
using Statistics
using Random

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
