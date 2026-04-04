import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import copy

# 1. Device Configuration (Optimized for M1 Mac)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Metal GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA GPU (CUDA)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# 2. Data Loading & Preprocessing
df = pd.read_excel("data1319.xlsx")

# Temporal split
train_df = df[df['year'] <= 2018].copy()
test_df  = df[df['year'] == 2019].copy()

def split_xy(data):
    X = data.drop(columns=['year', 'mo', 'model', 'share'])
    y = data['share'].values.reshape(-1, 1)
    return X, y

X_train_raw, y_train_raw = split_xy(train_df)
X_test_raw, y_test_raw = split_xy(test_df)

# Scaling
input_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_train_s = input_scaler.fit_transform(X_train_raw)
X_test_s  = input_scaler.transform(X_test_raw)
y_train_s = target_scaler.fit_transform(y_train_raw)
y_test_s  = target_scaler.transform(y_test_raw)

# Convert to Tensors
X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
y_train_t = torch.tensor(y_train_s, dtype=torch.float32)
X_test_t  = torch.tensor(X_test_s, dtype=torch.float32)
y_test_t  = torch.tensor(y_test_s, dtype=torch.float32)

# Create DataLoaders
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=16, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=16, shuffle=False)

# 3. Model Architecture
class DeepMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        return self.network(x)

# 4. Early Stopping Utility
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# 5. Training Logic
def train_model(model, train_loader, test_loader, epochs=300):
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()
    stopper = EarlyStopping(patience=25)
    history = {'train': [], 'val': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            outputs = model(bx)
            loss = criterion(outputs, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                val_loss += criterion(model(bx), by).item() * bx.size(0)
        
        avg_train = train_loss / len(train_loader.dataset)
        avg_val = val_loss / len(test_loader.dataset)
        history['train'].append(avg_train)
        history['val'].append(avg_val)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")
        
        stopper(avg_val, model)
        if stopper.early_stop:
            print(f"--> Early stopping at epoch {epoch+1}. Restoring best weights.")
            model.load_state_dict(stopper.best_model_state)
            break
            
    return history

# 6. Execution & Evaluation
model = DeepMLP(input_dim=X_train_t.shape[1]).to(device)
history = train_model(model, train_loader, test_loader)

# Final Metrics
model.eval()
with torch.no_grad():
    train_pred = model(X_train_t.to(device)).cpu().numpy()
    test_pred  = model(X_test_t.to(device)).cpu().numpy()

# Inverse scaling
train_pred_real = target_scaler.inverse_transform(train_pred)
test_pred_real  = target_scaler.inverse_transform(test_pred)
y_train_real    = target_scaler.inverse_transform(y_train_t.numpy())
y_test_real     = target_scaler.inverse_transform(y_test_t.numpy())

print("\n" + "="*30)
print(f"Final Train R2: {r2_score(y_train_real, train_pred_real):.4f}")
print(f"Final Test R2:  {r2_score(y_test_real, test_pred_real):.4f}")
print("="*30)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(history['train'], label='Train Loss')
plt.plot(history['val'], label='Val Loss (Test)')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()