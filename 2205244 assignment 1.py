import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalize(data):
    return (data - np.mean(data)) / np.std(data)

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = []
    for _ in range(iterations):
        gradient = (1 / m) * X.T.dot(X.dot(theta) - y)
        theta -= learning_rate * gradient
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

independent_data_path = 'path_to_independent_variable.csv'
dependent_data_path = 'path_to_dependent_variable.csv'

X = pd.read_csv(independent_data_path, header=None).values
y = pd.read_csv(dependent_data_path, header=None).values

X = normalize(X)

X = np.hstack((np.ones((X.shape[0], 1)), X))

theta = np.zeros((X.shape[1], 1))
learning_rate = 0.5
iterations = 50

theta, cost_history = gradient_descent(X, y, theta, learning_rate, iterations)

# Question 1
print("Final cost:", cost_history[-1])
print("Theta values:", theta)

# Question 3
plt.plot(range(1, iterations + 1), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function vs. Iterations')
plt.show()

# Question 4
plt.scatter(X[:, 1], y, color='blue', label='Data')
plt.plot(X[:, 1], X.dot(theta), color='red', label='Regression Line')
plt.xlabel('Independent Variable')
plt.ylabel('Dependent Variable')
plt.legend()
plt.show()

# Question 5
learning_rates = [0.005, 0.5, 5]
for lr in learning_rates:
    theta = np.zeros((X.shape[1], 1))
    _, cost_history = gradient_descent(X, y, theta, lr, iterations)
    plt.plot(range(1, iterations + 1), cost_history, label=f'LR={lr}')

plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function vs. Iterations for Different Learning Rates')
plt.legend()
plt.show()

# Question 6
def stochastic_gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = []
    for _ in range(iterations):
        for i in range(m):
            xi = X[i, :].reshape(1, -1)
            yi = y[i]
            gradient = xi.T.dot(xi.dot(theta) - yi)
            theta -= learning_rate * gradient
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

theta = np.zeros((X.shape[1], 1))
theta_sgd, cost_history_sgd = stochastic_gradient_descent(X, y, theta, learning_rate=0.05, iterations=iterations)

plt.plot(range(1, iterations + 1), cost_history_sgd, label='Stochastic GD')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function vs. Iterations (Stochastic GD)')
plt.legend()
plt.show()
