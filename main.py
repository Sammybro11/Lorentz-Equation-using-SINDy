import Models.TimeSeries_Models as TSModels
import numpy as np
from SINDy_Methods.LS_Method import Tseries_LS
import matplotlib.pyplot as plt


def test_LS():
    initial_coords = np.array([1,1])
    weights = np.array([[-1,0,2],[0.5, -2, 0]])
    Timeseries_data, dt = TSModels.Linear(initial_coords,weights,5, 100)
    Out_weights = Tseries_LS(Timeseries_data, dt)
    Predicted_data, dt = TSModels.Linear(initial_coords,Out_weights, 5, 100)
    print(f"True Equations: \n dx = {weights[0,0]} + {weights[0,1]}x +  {weights[0,2]}y \n",
          f"dy ={weights[1,0]} + {weights[1,1]}x +  {weights[1,2]}y ")
    print(f"Predicted Equations: \n dx = {Out_weights[0,0]:.2f} + {Out_weights[0,1]:.2f}x +  {Out_weights[0,2]:.2f}y \n",
          f"dy = {Out_weights[1,0]:.2f} + {Out_weights[1,1]:.2f}x +  {Out_weights[1,2]:.2f}y ")
    plt.figure()
    plt.plot(Timeseries_data[:,0], Timeseries_data[:,1], label='True')
    plt.plot(Predicted_data[:,0], Predicted_data[:,1], label='Predicted')
    plt.legend()
    plt.show()

test_LS()


