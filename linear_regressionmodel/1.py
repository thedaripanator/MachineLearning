import numpy as np
import matplotlib.pyplot as plt

# Generate 1000 training examples
np.random.seed(42)  # For reproducibility
X = 2 * np.random.rand(1000)  # Feature values between 0 and 2
true_m = 3.5                 # True slope
true_b = 1.2                 # True intercept
y = true_m * X + true_b + np.random.randn(1000) * 0.5  # Add noise

# Initialize parameters
m = 0
b = 0
learning_rate = 0.1
iterations = 10000
n = len(X)

# Function to compute Mean Squared Error (cost)
def compute_cost(X, y, m, b):
    y_pred = m * X + b
    return np.mean((y - y_pred)**2)

# Cost before gradient descent
initial_cost = compute_cost(X, y, m, b)
print(f"Initial cost: {initial_cost}")

# Store cost history
cost_history = []

# Gradient Descent
for i in range(iterations):
    y_pred = m * X + b
    dm = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)
    
    m = m - learning_rate * dm
    b = b - learning_rate * db
    
    cost = compute_cost(X, y, m, b)
    cost_history.append(cost)
    
    if i % 100 == 0:
        print(f"Iteration {i}: Cost = {cost:.4f}, m = {m:.4f}, b = {b:.4f}")

print(f"Trained model: y = {m:.4f}x + {b:.4f}")
print(f"Final cost: {cost_history[-1]:.4f}")
x1=float(input("Enter x:"))
y1=m*x1+b
print("Predicted y:",y1)

# Plot regression line and cost function
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(X, y, color='blue', alpha=0.5, s=10)
plt.plot(X, m*X + b, color='red')
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression Fit (1000 examples)")

plt.subplot(1,2,2)
plt.plot(range(iterations), cost_history, color='green')
plt.xlabel("Iteration")
plt.ylabel("Cost (MSE)")
plt.title("Cost Function Reduction")
plt.show()