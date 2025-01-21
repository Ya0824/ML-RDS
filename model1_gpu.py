import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    filename="gpu_model_training.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# Load data
logger.info("Loading data")
data = np.load("concat.npy")
np.random.shuffle(data)

# Split data into training and testing
logger.info("Splitting data")
train = data[:1000000]
test = data[1000000:]
X_train = train[:, :-1]
Y_train = train[:, -1]
X_test = test[:, :-1]
Y_test = test[:, -1]

# Normalize data
logger.info("Normalizing data")
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train.reshape(-1, 1)).ravel()
Y_test = scaler_Y.transform(Y_test.reshape(-1, 1)).ravel()

# Convert to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).to(device)

# Define the neural network
logger.info("Defining the neural network")
class MLPModel(nn.Module):
    def __init__(self, input_size):
        super(MLPModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

model = MLPModel(X_train.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
logger.info("Training the model")
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train_tensor).squeeze()
    loss = criterion(predictions, Y_train_tensor)
    loss.backward()
    optimizer.step()
    logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Evaluate the model
logger.info("Evaluating the model")
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).squeeze()
    test_loss = criterion(predictions, Y_test_tensor).item()
    predictions_inverse = scaler_Y.inverse_transform(predictions.cpu().numpy().reshape(-1, 1)).ravel()
    Y_test_inverse = scaler_Y.inverse_transform(Y_test_tensor.cpu().numpy().reshape(-1, 1)).ravel()
    r2 = r2_score(Y_test_inverse, predictions_inverse)
    rmse = np.sqrt(mean_squared_error(Y_test_inverse, predictions_inverse))

logger.info(f"Test Loss: {test_loss:.4f}")
logger.info(f"R^2: {r2:.2f}")
logger.info(f"RMSE: {rmse:.2f}")

print(f"R^2: {r2:.2f}")
print(f"RMSE: {rmse:.2f}")

