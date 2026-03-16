import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

# -----------------------------
# 1 Load Dataset
# -----------------------------

data = pd.read_csv("kc_house_data.csv")

features = ['sqft_living','bedrooms','bathrooms','floors','yr_built']
X_train = data[features].values
y_train = data['price'].values

print("Dataset Loaded")
print("X shape:", X_train.shape)
print("y shape:", y_train.shape)

# -----------------------------
# 2 Feature Scaling
# -----------------------------

X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)

X_train = (X_train - X_mean) / X_std

# -----------------------------
# 3 Cost Function
# -----------------------------

def compute_cost(X, y, w, b):

    m = X.shape[0]
    cost = 0

    for i in range(m):

        f_wb = np.dot(X[i], w) + b
        cost += (f_wb - y[i])**2

    cost = cost/(2*m)
    return cost


# -----------------------------
# 4 Gradient Calculation
# -----------------------------

def compute_gradient(X, y, w, b):

    m,n = X.shape

    dj_dw = np.zeros(n)
    dj_db = 0

    for i in range(m):

        error = (np.dot(X[i],w) + b) - y[i]

        for j in range(n):
            dj_dw[j] += error * X[i,j]

        dj_db += error

    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_db, dj_dw


# -----------------------------
# 5 Gradient Descent
# -----------------------------

def gradient_descent(X, y, w_in, b_in, alpha, iterations):

    w = copy.deepcopy(w_in)
    b = b_in

    J_history = []

    for i in range(iterations):

        dj_db, dj_dw = compute_gradient(X,y,w,b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = compute_cost(X,y,w,b)
        J_history.append(cost)

        if i % (iterations//10) == 0:
            print("Iteration:", i, "Cost:", cost)

    return w,b,J_history


# -----------------------------
# 6 Initialize Parameters
# -----------------------------

n = X_train.shape[1]

initial_w = np.zeros(n)
initial_b = 0

alpha = 0.01
iterations = 5000

# -----------------------------
# 7 Initial Cost (w=0, b=0)
# -----------------------------

initial_cost = compute_cost(X_train, y_train, initial_w, initial_b)

print("\nInitial Cost when w=0 and b=0:", initial_cost)

# -----------------------------
# 8 Train Model
# -----------------------------

w_final, b_final, J_hist = gradient_descent(
    X_train,
    y_train,
    initial_w,
    initial_b,
    alpha,
    iterations
)

print("\nTraining Complete")
print("Final weights:", w_final)
print("Final bias:", b_final)


# -----------------------------
# 9 Predictions
# -----------------------------

print("\nSample Predictions\n")

for i in range(5):

    prediction = np.dot(X_train[i],w_final) + b_final
    print("Predicted:", round(prediction,2),
          "Actual:", y_train[i])


# -----------------------------
# 10 Predict New House Price
# -----------------------------

new_house = np.array([2000,3,2,2,2005])

# feature scaling for input
new_house = (new_house - X_mean)/X_std

price = np.dot(new_house,w_final) + b_final

print("\nPredicted price for new house:", round(price,2))


# -----------------------------
# 11 Graph: Cost vs Iterations
# -----------------------------

plt.figure(figsize=(8,5))
plt.plot(J_hist)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs Iterations")
plt.show()


# -----------------------------
# 12 Graph: Training Data + Regression Line
# -----------------------------

x_vis = data['sqft_living'].values
y_vis = data['price'].values

x_vis_norm = (x_vis - X_mean[0]) / X_std[0]

y_pred_line = w_final[0] * x_vis_norm + b_final

plt.figure(figsize=(8,5))

plt.scatter(x_vis, y_vis, alpha=0.3, label="Training Data")

sorted_index = np.argsort(x_vis)

plt.plot(x_vis[sorted_index],
         y_pred_line[sorted_index],
         color='red',
         linewidth=3,
         label="Regression Line")

plt.xlabel("sqft_living")
plt.ylabel("price")
plt.title("House Price vs Living Area")
plt.legend()

plt.show()