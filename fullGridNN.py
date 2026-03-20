import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

import itertools
import copy

# -----------------------------
# Temporal split (UPDATED)
# -----------------------------
train_df = df[df['year'] <= 2017].copy()
val_df   = df[df['year'] == 2018].copy()
test_df  = df[df['year'] == 2019].copy()

def split_xy(data):
    X = data.drop(columns=['year', 'mo', 'model', 'share'])
    y = data['share'].values.reshape(-1, 1)
    return X, y

X_train, y_train = split_xy(train_df)
X_val, y_val     = split_xy(val_df)
X_test, y_test   = split_xy(test_df)

# -----------------------------
# Scaling
# -----------------------------
input_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_train = input_scaler.fit_transform(X_train)
X_val   = input_scaler.transform(X_val)
X_test  = input_scaler.transform(X_test)

y_train = target_scaler.fit_transform(y_train)
y_val   = target_scaler.transform(y_val)
y_test  = target_scaler.transform(y_test)

# Convert to tensors
def to_tensor(x, y):
    return torch.FloatTensor(x), torch.FloatTensor(y)

X_train_t, y_train_t = to_tensor(X_train, y_train)
X_val_t, y_val_t     = to_tensor(X_val, y_val)
X_test_t, y_test_t   = to_tensor(X_test, y_test)

# -----------------------------
# Model definition
# -----------------------------
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, activation):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# -----------------------------
# Training function
# -----------------------------
def train_model(params):
    model = MLP(
        input_size=X_train_t.shape[1],
        hidden_size=params['hidden_size'],
        dropout=params['dropout'],
        activation=params['activation']
    )
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=params['lr'],
        weight_decay=params['weight_decay']
    )
    
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_model = None
    patience = 10
    counter = 0
    
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
    
    return best_model, best_val_loss.item()

# -----------------------------
# Grid Search
# -----------------------------
param_grid = {
    'hidden_size': [5, 10, 20],
    'lr': [0.001, 0.005],
    'dropout': [0.0, 0.2],
    'weight_decay': [0.0, 1e-4],
    'activation': ['relu', 'leaky_relu']
}

keys, values = zip(*param_grid.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

best_score = float('inf')
best_params = None
best_model = None

for params in experiments:
    model, val_loss = train_model(params)
    
    print(f"Params: {params}, Val Loss: {val_loss:.6f}")
    
    if val_loss < best_score:
        best_score = val_loss
        best_params = params
        best_model = model

print("\nBest Params:", best_params)

# -----------------------------
# Final evaluation
# -----------------------------
best_model.eval()
with torch.no_grad():
    test_pred = best_model(X_test_t).numpy()

test_pred = target_scaler.inverse_transform(test_pred)
test_r2 = r2_score(target_scaler.inverse_transform(y_test), test_pred)

print("Final Test R2:", test_r2)