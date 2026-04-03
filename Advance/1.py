import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Normalization
import matplotlib.pyplot as plt

# 1. CREATE SYNTHETIC KAGGLE-STYLE DATA
# Let's generate 200 samples of [Temperature, Duration]
def load_coffee_data():
    np.random.seed(42)
    # Temps between 150-285°C, Durations between 11.5-15.5 mins
    X = np.random.rand(200, 2)
    X[:,0] = X[:,0] * (285-150) + 150 
    X[:,1] = X[:,1] * (15.5-11.5) + 11.5
    
    # Define "Good Roast" (y=1) as: 
    # Temp between 175-260 AND Duration between 12-15
    Y = np.zeros((200,1))
    for i in range(200):
        if (175 <= X[i,0] <= 260) and (12 <= X[i,1] <= 15):
            Y[i] = 1
    return X, Y

X_train, y_train = load_coffee_data()

# 2. DATA NORMALIZATION
# Neural networks struggle when one feature (Temp ~200) is 
# much larger than another (Time ~13). We scale them to a similar range.
norm_layer = Normalization(axis=-1)
norm_layer.adapt(X_train)
X_train_n = norm_layer(X_train)

# 3. DEFINE THE NEURON (LOGISTIC REGRESSION)
# A single Dense unit with 'sigmoid' is a Logistic Regression neuron.
model = Sequential([
    tf.keras.Input(shape=(2,)),
    norm_layer,
    Dense(units=1, activation='sigmoid', name='logistic_layer')
])

# 4. COMPILE AND TRAIN
# We use Binary Crossentropy because we are classifying (Good vs Bad)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='binary_crossentropy'
)

print("Training started...")
history = model.fit(X_train, y_train, epochs=200, verbose=0)
print("Training finished.")

# 5. TEST THE MODEL
# Test 1: Should be a Good Roast (1)
test_good = np.array([[200.0, 13.5]]) 
# Test 2: Should be a Bad Roast (0) - Too hot
test_bad = np.array([[280.0, 14.0]]) 

pred_good = model.predict(test_good)
pred_bad = model.predict(test_bad)

print(f"\nPrediction for 200°C @ 13.5 mins: {pred_good[0][0]:.4f} (Likely Good)")
print(f"Prediction for 280°C @ 14.0 mins: {pred_bad[0][0]:.4f} (Likely Bad)")

# 6. VIEW LEARNED WEIGHTS
w, b = model.get_layer("logistic_layer").get_weights()
print(f"\nLearned Weights (w): {w}")
print(f"Learned Bias (b): {b}")