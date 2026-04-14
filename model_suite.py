import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# --- 1. Device Setup (M1 Mac Optimization) ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- 2. Load and Split Dataset ---
df = pd.read_excel("data1319.xlsx")
train_df = df[df['year'] <= 2018].copy()
test_df  = df[df['year'] == 2019].copy()

y_train = train_df['share'].values
y_test  = test_df['share'].values

exclude_cols = ["share", "mo", "year", "model"]
X_train_raw = train_df.drop(columns=exclude_cols)
X_test_raw  = test_df.drop(columns=exclude_cols)

# --- 3. Encoding & Scaling ---
X_train, X_test = X_train_raw.copy(), X_test_raw.copy()
for col in X_train.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col]  = le.transform(X_test[col])

in_scaler = MinMaxScaler()
out_scaler = MinMaxScaler()

X_train_scaled = torch.tensor(in_scaler.fit_transform(X_train), dtype=torch.float32)
X_test_scaled  = torch.tensor(in_scaler.transform(X_test), dtype=torch.float32)
y_train_scaled = torch.tensor(out_scaler.fit_transform(y_train.reshape(-1, 1)), dtype=torch.float32)
y_test_scaled  = torch.tensor(out_scaler.transform(y_test.reshape(-1, 1)), dtype=torch.float32)

# --- 4. Model Architectures ---

# Standard MLP
class DeepMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32), nn.LeakyReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.LeakyReLU(), nn.Dropout(0.1),
            nn.Linear(16, 1)
        )
    def forward(self, x): return self.net(x)

# Integrated LSTM/GRU Architecture
class RecurrentNet(nn.Module):
    def __init__(self, input_size, hidden_size=32, model_type="LSTM"):
        super().__init__()
        if model_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class EarlyStopping:
    def __init__(self, patience=20):
        self.patience, self.counter, self.best_loss = patience, 0, float('inf')
        self.early_stop, self.best_state = False, None
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss, self.best_state = val_loss, copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True

# --- 5. Training tree-based Models ---
tscv = TimeSeriesSplit(n_splits=4)

print("Training XGBoost...")
xgb_grid = GridSearchCV(XGBRegressor(random_state=42), 
                        param_grid={"n_estimators": [300, 500], "max_depth": [3, 5]}, 
                        cv=tscv, scoring="r2", n_jobs=-1).fit(X_train, y_train)

print("Training Random Forest...")
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), 
                       param_grid={"n_estimators": [200, 400], "max_features": ["sqrt"]}, 
                       cv=tscv, scoring="r2", n_jobs=-1).fit(X_train, y_train)

# --- 6. Training Neural Networks (MLP, LSTM, GRU) ---
criterion = nn.MSELoss()

def train_torch_model(model, train_x, train_y, test_x, test_y):
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    stopper = EarlyStopping()
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=16, shuffle=True)
    
    for epoch in range(300):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(); loss = criterion(model(bx), by); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(test_x.to(device)), test_y.to(device)).item()
        stopper(val_loss, model)
        if stopper.early_stop: break
    model.load_state_dict(stopper.best_state)
    return model

print("Training Neural Network (MLP)...")
nn_model = train_torch_model(DeepMLP(X_train_scaled.shape[1]).to(device), 
                             X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)

# RNNs need 3D input: (Batch, Sequence Length=1, Features)
X_tr_rnn, X_te_rnn = X_train_scaled.unsqueeze(1), X_test_scaled.unsqueeze(1)

print("Training LSTM...")
lstm_model = train_torch_model(RecurrentNet(X_train_scaled.shape[1], model_type="LSTM").to(device),
                               X_tr_rnn, y_train_scaled, X_te_rnn, y_test_scaled)

print("Training GRU...")
gru_model = train_torch_model(RecurrentNet(X_train_scaled.shape[1], model_type="GRU").to(device),
                              X_tr_rnn, y_train_scaled, X_te_rnn, y_test_scaled)

# --- 7. Unified Evaluation ---
def get_full_metrics(name, model_obj, X_tr, y_tr, X_te, y_te, is_nn=False):
    if is_nn:
        model_obj.eval()
        with torch.no_grad():
            train_p = out_scaler.inverse_transform(model_obj(X_tr.to(device)).cpu().numpy()).flatten()
            test_p = out_scaler.inverse_transform(model_obj(X_te.to(device)).cpu().numpy()).flatten()
    else:
        train_p = model_obj.predict(X_tr)
        test_p = model_obj.predict(X_te)
    
    return [
        {"Model": name, "Dataset": "Train", "R2": r2_score(y_tr, train_p), "RMSE": np.sqrt(mean_squared_error(y_tr, train_p))},
        {"Model": name, "Dataset": "Test",  "R2": r2_score(y_te, test_p),  "RMSE": np.sqrt(mean_squared_error(y_te, test_p))}
    ]

all_results = []
all_results.extend(get_full_metrics("XGBoost", xgb_grid.best_estimator_, X_train, y_train, X_test, y_test))
all_results.extend(get_full_metrics("Random Forest", rf_grid.best_estimator_, X_train, y_train, X_test, y_test))
all_results.extend(get_full_metrics("Neural Network", nn_model, X_train_scaled, y_train, X_test_scaled, y_test, is_nn=True))
all_results.extend(get_full_metrics("LSTM", lstm_model, X_tr_rnn, y_train, X_te_rnn, y_test, is_nn=True))
all_results.extend(get_full_metrics("GRU", gru_model, X_tr_rnn, y_train, X_te_rnn, y_test, is_nn=True))

# --- 8. Results Display & Visualization ---
results_df = pd.DataFrame(all_results)

print("\n" + "="*60)
print("FINAL PERFORMANCE COMPARISON: ALL MODELS")
print("="*60)
for model_name in ["Random Forest", "XGBoost", "Neural Network", "LSTM", "GRU"]:
    subset = results_df[results_df["Model"] == model_name]
    print(subset.to_string(index=False))
    print("-" * 60)

# Visual Comparison of Test R2
plt.figure(figsize=(10, 5))
test_comparison = results_df[results_df["Dataset"] == "Test"].sort_values(by="R2", ascending=False)
colors = plt.cm.viridis(np.linspace(0, 1, len(test_comparison)))
plt.bar(test_comparison["Model"], test_comparison["R2"], color=colors)
plt.title("Final Test R2 Comparison (All Models)")
plt.ylabel("R2 Score")
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Feature Importance (Based on original winning model: RF)
best_rf = rf_grid.best_estimator_
feature_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': best_rf.feature_importances_}).sort_values(by='Importance', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(feature_df['Feature'], feature_df['Importance'], color='#2c3e50')
plt.title("Key Drivers of Market Share (Random Forest Importance)")
plt.xlabel("Gini Importance Score")
plt.tight_layout()
plt.show()