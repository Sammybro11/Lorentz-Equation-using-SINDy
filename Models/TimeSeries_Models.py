import numpy as np

def Linear(init, weights, t_max, num):
    """
    Creates time series data for
    x_dot = c + ax + by
    y_dot = c + ax + by
    :param init: [x, y]
    :param weights: [[x_dot coefficients], [y_dot coefficients]]
    :param t_max:
    :param num: number of time points
    :return:
    """

    t_series = np.linspace(0,t_max,num)
    dt = t_series[1] - t_series[0]
    m = len(t_series)
    n = len(init)
    out = np.zeros((m,n))
    out[0,:] = init
    for i in range(1,m):
        state_prev = out[i-1,:]
        dx = weights[0,0] + weights[0,1]*state_prev[0] + weights[0,2]*state_prev[1]
        dy = weights[1,0] + weights[1,1]*state_prev[0] + weights[1,2]*state_prev[1]
        out[i,0] = state_prev[0] + dx*dt
        out[i,1] = state_prev[1] + dy*dt
    return out, dt
