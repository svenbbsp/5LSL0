# %% imports
# libraries
import numpy as np
from numpy import transpose as T
from numpy.linalg import inv 

# %% exercise 3
# input parameters
X = np.array([[0.0, 0.1, 1.0, 1.0],
              [0.0, 1.0, 0.2, 1.0],
              [1.0, 1.0, 1.0, 1.0]])

y =  np.array([0, 0.41, 0.18, 0.5])

# calculate optimal theta
theta = inv(X @ T(X)) @ X @ T(y)

# print results
print("----------------------")
print("For exercise 3 we get:")
print(f"w1  = {theta[0]:.4}")
print(f"w2  = {theta[1]:.4}")
print(f"b   = {theta[2]:.4}")

# calculate the new loss.
mse = ((y - T(theta) @ X)**2).mean()

print(f"mse = {mse:.4}")
print("")

# %% exercise 4
# change the measurements y    
y_new = np.array([-0.0416, 0.3610, 0.1222, 0.4733])

# calculate optimal theta
theta_new = inv(X @ T(X)) @ X @ T(y_new)

# print results
print("----------------------")
print("For exercise 4 we get:")
print(f"w1  = {theta_new[0]:.4}")
print(f"w2  = {theta_new[1]:.4}")
print(f"b   = {theta_new[2]:.4}")

# calculate the new loss.
mse_new = ((y_new - T(theta_new) @ X)**2).mean()

print(f"mse = {mse_new:.4}")
print("")

# %% exercise 6
# change the inputs X and measurements y
X_xor = np.array([[0.0, 0.0, 1.0, 1.0],
                  [0.0, 1.0, 0.0, 1.0],
                  [1.0, 1.0, 1.0, 1.0]])

y_xor =  np.array([0.0, 1.0, 1.0, 0.0])

# calculate optimal theta
theta_xor = inv(X_xor @ T(X_xor)) @ X_xor @ T(y_xor)

# print results
print("----------------------")
print("For exercise 6 we get:")
print(f"w1  = {theta_xor[0]:.4}")
print(f"w2  = {theta_xor[1]:.4}")
print(f"b   = {theta_xor[2]:.4}")

# calculate the new loss.
mse_xor = ((y_xor - T(theta_xor) @ X_xor)**2).mean()

print(f"mse = {mse_new:.4}")
print("")