import numpy as np


def polynomial_libarybuilder(X: np.ndarray, degree: int = 1):
    N, d = X.shape

    Library = np.hstack((np.ones((N,1), dtype= float), X))

    if degree >= 2:
        for i in range(d):
            for j in range(d):
                column = X[:,i] * X[:,j]
                column = np.reshape(column, (N,1))
                Library = np.hstack((Library, column))

    return Library


X = np.array([[2,3],
    [2,3] ])

print(polynomial_libarybuilder(X, degree = 2))