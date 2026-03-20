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
df = pd.read_excel("data1319.xlsx")

# -----------------------------
# 2. Temporal train-test split
# -----------------------------
train_df = df[df['year'] <= 2018].copy()
test_df  = df[df['year'] == 2019].copy()

def split_xy(data):
    X = data.drop(columns=['year', 'mo', 'model', 'share'])
    y = data['share'].values.reshape(-1, 1)
    return X, y

X_train, y_train = split_xy(train_df)
X_test, y_test   = split_xy(test_df)

# -----------------------------
# 3. Scaling
# -----------------------------
input_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_train = input_scaler.fit_transform(X_train)
X_test  = input_scaler.transform(X_test)

y_train = target_scaler.fit_transform(y_train)
y_test  = target_scaler.transform(y_test)

# Convert to tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_test_t  = torch.FloatTensor(X_test)
y_test_t  = torch.FloatTensor(y_test)

# -----------------------------
# 4. Neural Network Definition
# -----------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size=10, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

input_size = X_train_t.shape[1]
model = SimpleMLP(input_size=input_size, hidden_size=10, dropout=0.2)

# -----------------------------
# 5. Loss and Optimizer
# -----------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# -----------------------------
# 6. Training Loop
# -----------------------------
num_epochs = 150
batch_size = 10
history = []

for epoch in range(num_epochs):
    permutation = torch.randperm(X_train_t.size(0))
    epoch_loss = 0.0
    
    for i in range(0, X_train_t.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train_t[indices], y_train_t[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * batch_x.size(0)
        
    epoch_loss /= X_train_t.size(0)
    history.append(epoch_loss)
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")

# -----------------------------
# 7. Predictions & Performance
# -----------------------------
model.eval()
with torch.no_grad():
    train_pred = model(X_train_t).numpy()
    test_pred  = model(X_test_t).numpy()

train_pred = target_scaler.inverse_transform(train_pred)
test_pred  = target_scaler.inverse_transform(test_pred)

train_r2 = r2_score(target_scaler.inverse_transform(y_train), train_pred)
test_r2  = r2_score(target_scaler.inverse_transform(y_test), test_pred)

print("\nPerformance Results:")
print("Training R2:", train_r2)
print("Testing R2:", test_r2)

# -----------------------------
# 8. Plot training loss
# -----------------------------
plt.figure(figsize=(8,6))
plt.plot(history, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Training Loss')
plt.legend()
plt.show()