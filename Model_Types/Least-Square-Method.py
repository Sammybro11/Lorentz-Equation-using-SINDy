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

def CF_differentiate(X, dt):
    """
    Computes X_dot using Centered Finite Difference Method.
    :param X: [m x n] matrix for m time series and n states
    :param dt: time step
    :return: X_dot: Derivative of X
    """

    X_dot = np.zeros_like(X)
    X_dot[0, :] = (X[0, :] - X[1, :])/dt
    X_dot[1:-1, :] = (X[0:-2, :] - X[2:, :])/dt
    X_dot[-1, :] = (X[-1, :] - X[-2, :])/dt
    return X_dot

def Tseries_LS(X: np.ndarray, dt: float, differentiator):
    """
    Calculate the Linear Model weights for a Time series data using Least Squares Method.

    :param X: [m x n] matrix for m time series and n states
    :param differentiator: function used to compute derivative of X
    :param dt: time step
    :return:
    """
    # Standardizing X
    means = np.mean(X, axis=0)
    X_c = X - means
    scales = X_c.std(axis=0)
    scales[scales ==0] = eps
    X = X_c / scales

    # Create Library
    X_dot = differentiator(X, dt)
    Library = np.hstack([np.ones(X.shape[0], dtype=float), X])
    # Solve using Pseudoinverse
    weights = np.linalg.pinv(Library) @ X_dot

    # Rescale weights

    weights = (weights * scales)



