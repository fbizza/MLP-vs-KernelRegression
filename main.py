import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

np.random.seed(29)

# Generate x data
x_true = np.linspace(-20, 20, 1000).reshape(-1, 1)
x_sample = np.sort(np.random.uniform(low=-20, high=20, size=500)).reshape(-1, 1)
x_test = np.linspace(-20, 20, 200).reshape(-1, 1)

# Compute y data
y_true = np.sin(x_true) * (1 + x_true / 2)
y_sample = np.sin(x_sample) * (1 + x_sample / 2)

# Create and fit Multilayer perceptron
mlp = MLPRegressor(hidden_layer_sizes=(50, 100, 200, 100, 50), activation='relu', solver='adam', max_iter=300)
mlp.fit(x_sample, y_sample)

# Predict y values for x data
y_test = mlp.predict(x_test)

# Plot the results
plt.scatter(x_sample, y_sample, color='red', s=6, label='Samples')
plt.plot(x_true, y_true, color='blue', label='True function')
plt.plot(x_test, y_test, color='orange', label='MLP fit')
plt.legend()
plt.show()

