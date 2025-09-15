# Constructing Least Squares Method
# Implements Identification of Non-Linear Dynamics without Sparsity

import numpy as np
from SINDy_Methods.helper_functions import polynomial_libarybuilder

eps = 1e-6

def CF_differentiate(X, dt):
    """
    Computes X_dot using Centered Finite Difference Method.
    :param X: [m x n] matrix for m time series and n states
    :param dt: time step
    :return: X_dot: Derivative of X
    """

    X_dot = np.zeros_like(X)
    X_dot[0, :] = (X[1, :] - X[0, :])/dt
    X_dot[1:-1, :] = (X[2:, :] - X[:-2,:])/(2*dt)
    X_dot[-1, :] = (X[-1, :] - X[-2, :])/dt
    return X_dot

def Tseries_LS(X: np.ndarray, dt: float, differentiator = CF_differentiate, linearity =  True, degree = 1):
    """
    Calculate the Linear Model weights for a Time series data using Least Squares Method.

    :param X: [m x n] matrix for m time series and n states
    :param dt: time step
    :param differentiator: function used to compute derivative of X
    :param linearity: Boolean to determine if the library should be built with non-linear terms
    :param degree: Degree of non-linear terms
    :return: weights_raw
    """
    # Standardizing X

    means = np.mean(X, axis=0)
    X_c = X - means
    scales = X_c.std(axis=0)
    scales[scales ==0] = eps
    X = X_c / scales

    # Create Library
    X_dot = differentiator(X, dt)
    if linearity:
        Library = polynomial_libarybuilder(X, degree = 1)
    else:
        Library = polynomial_libarybuilder(X, degree = 2)
        print(Library)
    # Solve using Pseudoinverse
    weights = np.linalg.pinv(Library[1:-1,:]) @ X_dot[1:-1,:]
    print(weights)
    if linearity:
        # Rescale weights

        intercept_std = weights[0, :]  # (n,)
        coeffs_std = weights[1:, :].T  # (n×n): columns are equations

        D = np.diag(scales)  # (n×n)
        D_inv = np.diag(1.0 / scales)  # (n×n)
        # This part is wrong in non-linear fix it gng
        coeffs_raw = D @ coeffs_std @ D_inv  # (n×n)
        intercept_raw = (D @ intercept_std) - (coeffs_raw @ means)  # (n,)

        weights_raw = np.zeros((coeffs_raw.shape[0], coeffs_raw.shape[1] + 1), dtype=float)
        weights_raw[:, 0] = intercept_raw
        weights_raw[:, 1:] = coeffs_raw

        return weights_raw
    else:
        return weights





