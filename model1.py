import pickle
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Loading data
print("Loading data")
data = np.load("concat.npy")
print(data.shape)
np.random.shuffle(data)

# Divide the dataset into training and testing sets
print("Divide the dataset")
train = data[:1000000]
test = data[1000000:]
np.save("model_train", train)
np.save("model_test", test)
print("Data split completed")
X = train[:, :-1]
Y = train[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Data normalization
print("Data normalization")
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train.reshape(-1, 1)).ravel()
Y_test = scaler_Y.transform(Y_test.reshape(-1, 1)).ravel()
print("Data normalization completed")

# MLP model
model = MLPRegressor(
    hidden_layer_sizes=(128, 256, 128, 64, 1),  # More deeper layers
    activation='relu',                          
    solver='adam',                              
    alpha=1e-4,                                 
    learning_rate_init=0.001,                   
    max_iter=500,                               
    early_stopping=True,                        
    verbose=True,                               
    random_state=42                             
)

# Model training
print("Model training")
model.fit(X_train, Y_train)
print("Model training completed")
# Save the model
print("Model saving")
model_bundle = {
    "model": model,
    "scaler_X": scaler_X,
    "scaler_Y": scaler_Y
}
with open("model_bundle.pickle", "wb") as f:
    pickle.dump(model_bundle, f)
print("Model saving completed")

# Model prediction and evaluation
print("Model prediction and evaluation")
predictions = model.predict(X_test)
print("Model prediction completed")
# Inverse normalized prediction results
print("Inverse normalized prediction results")
Y_test_inverse = scaler_Y.inverse_transform(Y_test.reshape(-1, 1)).ravel()
predictions_inverse = scaler_Y.inverse_transform(predictions.reshape(-1, 1)).ravel()

r2 = r2_score(Y_test_inverse, predictions_inverse)
rmse = np.sqrt(mean_squared_error(Y_test_inverse, predictions_inverse))

print(f"R^2: {r2:.2f}")
print(f"RMSE: {rmse:.2f}")
