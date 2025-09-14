# Constructing Least Squares Method
# Implements Identification of Non-Linear Dynamics without Sparsity

'''
Inputs:
  - Time series X ∈ R^{m×n}, sampled every Δt
  - Numerical differentiation routine to compute Xdot from X
Procedure:
  1) Compute Xdot = differentiate(X, Δt)  // centered difference or TV-regularized
  2) Build library Θ = concat_columns(ones(m,1), X)  // linear + intercept
  3) Optionally standardize Θ columns except the intercept for conditioning
  4) For j in 1..n:
       ξ_j = least_squares(Θ, Xdot[:, j])  // pseudoinverse or stable solver
  5) Stack Xi = [ξ_1, …, ξ_n]  // shape (1+n)×n
  6) Extract b = Xi[0, :], A = Xi[1:, :]
Outputs:
  - Linear model: ẋ = b + A x
Validation:
  - Simulate ẋ = b + A x from held-out initial states and compare to data
'''