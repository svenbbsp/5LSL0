# %% imports
# libraries
import numpy as np
from numpy import transpose as T
from numpy.linalg import inv 
import matplotlib.pyplot as plt

# %% exercise 10
# Network parameters of the first layer
W_1 = np.ones((2,2))
b_1 = np.array([[0],[-1]])
w_2 = np.array([[1],[-2]])
b_2 = 0

# inputs
X = np.array([[0.0, 0.0, 1.0, 1.0],
              [0.0, 1.0, 0.0, 1.0],])

y = np.array([0.0, 1.0, 1.0, 0.0])

# map to latent space:
h = W_1 @ X + b_1 # weights and bias
h = h * (h>0)     # ReLU

# plot the points in latent space
plt.figure()
plt.scatter(h[0,y==1],h[1,y==1],marker='x')
plt.scatter(h[0,y==0],h[1,y==0],marker='o')

# plotting the decision boundary
h_1 = np.array([-1,3])
h_2 = -0.25 + 0.5 * h_1
plt.plot(h_1,h_2,'--',c='k')

# beautification of the plot
plt.xlim(-0.2,2.2)
plt.ylim(-0.2,1.2)
plt.xlabel('h1')
plt.ylabel('h2')
ax = plt.gca()
ax.set_aspect('equal')
plt.title('Latent Space')
plt.savefig("figures//exercise_10.png", dpi=300, bbox_inches = 'tight')
plt.show()

# %% exercise 11
# plot the points in latent space
plt.figure()
plt.scatter(X[0,y==1],X[1,y==1],marker='x')
plt.scatter(X[0,y==0],X[1,y==0],marker='o')

# plotting the decision boundary
x_1 = np.array([-2,2])
x_2_rule1 = 1.5 - x_1
x_2_rule2 = 0.5 - x_1

plt.plot(x_1,x_2_rule1,'--',c='k')
plt.plot(x_1,x_2_rule2,'--',c='k')

# shading
x_2_shading = 1 - x_1
plt.fill_between(x_1,  [-10, -10], x_2_shading, alpha = 0.2)
plt.fill_between(x_1, x_2_shading,    [10, 10], alpha = 0.2)

# beautification of the plot
plt.xlim(-0.2,1.2)
plt.ylim(-0.2,1.2)
plt.xlabel('x1')
plt.ylabel('x2')
ax = plt.gca()
ax.set_aspect('equal')
plt.title('Input Space')
plt.savefig("figures//exercise_11.png", dpi=300, bbox_inches = 'tight')
plt.show()
