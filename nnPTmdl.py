import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_excel('data1319.xlsx', sheet_name=0)

# -----------------------------
# 2. Temporal train/test split
# -----------------------------
train_df = df[df['year'] <= 2018].copy()
test_df  = df[df['year'] == 2019].copy()

# Separate features and target
X_train = train_df.drop(columns=['year', 'mo', 'model', 'share'])
y_train = train_df['share'].values.reshape(-1, 1)

X_test = test_df.drop(columns=['year', 'mo', 'model', 'share'])
y_test = test_df['share'].values.reshape(-1, 1)

# -----------------------------
# 3. Preprocessing (scaling)
# -----------------------------
input_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_train_scaled = input_scaler.fit_transform(X_train)
X_test_scaled  = input_scaler.transform(X_test)

y_train_scaled = target_scaler.fit_transform(y_train)
y_test_scaled  = target_scaler.transform(y_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train_scaled)
X_test_tensor  = torch.FloatTensor(X_test_scaled)
y_test_tensor  = torch.FloatTensor(y_test_scaled)

# -----------------------------
# 4. Define the neural network
# -----------------------------
class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_size = X_train_tensor.shape[1]
hidden_size = 10
output_size = y_train_tensor.shape[1]

model = MLPRegressor(input_size, hidden_size, output_size)

# -----------------------------
# 5. Loss and optimizer
# -----------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# -----------------------------
# 6. Training loop
# -----------------------------
num_epochs = 100
batch_size = 10
history = {'loss': []}

for epoch in range(num_epochs):
    permutation = torch.randperm(X_train_tensor.size()[0])
    epoch_loss = 0.0
    
    for i in range(0, X_train_tensor.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * batch_x.size(0)
        
    epoch_loss /= X_train_tensor.size(0)
    history['loss'].append(epoch_loss)
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")

# -----------------------------
# 7. Predictions
# -----------------------------
with torch.no_grad():
    train_pred_scaled = model(X_train_tensor).numpy()
    test_pred_scaled  = model(X_test_tensor).numpy()

train_pred = target_scaler.inverse_transform(train_pred_scaled)
test_pred  = target_scaler.inverse_transform(test_pred_scaled)

# -----------------------------
# 8. Performance evaluation
# -----------------------------
train_mse = np.mean((train_pred_scaled - y_train_scaled)**2)
test_mse  = np.mean((test_pred_scaled - y_test_scaled)**2)
target_variance = np.var(y_train_scaled)

train_accuracy = max(0, 1 - (train_mse / target_variance))
test_accuracy  = max(0, 1 - (test_mse / target_variance))

train_r2 = r2_score(y_train, train_pred)
test_r2  = r2_score(y_test, test_pred)

print("\nPerformance Results:")
print("Test MSE:", test_mse)
print("Adjusted Training Accuracy:", train_accuracy)
print("Adjusted Testing Accuracy:", test_accuracy)
print("Training R-squared:", train_r2)
print("Testing R-squared:", test_r2)

# -----------------------------
# 9. Plot training loss
# -----------------------------
plt.figure(figsize=(8,6))
plt.plot(history['loss'], label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Training Loss')
plt.legend()
plt.show()

# -----------------------------
# 10. Deployment function
# -----------------------------
def my_neural_network_function(new_X):
    new_X_scaled = input_scaler.transform(new_X)
    new_X_tensor = torch.FloatTensor(new_X_scaled)
    with torch.no_grad():
        y_scaled = model(new_X_tensor).numpy()
    return target_scaler.inverse_transform(y_scaled)