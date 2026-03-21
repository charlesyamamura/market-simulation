import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
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

# -----------------------------
# 4. Neural Network Definition
# -----------------------------
input_size = X_train.shape[1]
hidden_size = 8
dropout_rate = 0.2

model = Sequential([
    Dense(hidden_size, input_shape=(input_size,)),
    LeakyReLU(),
    Dropout(dropout_rate),
    Dense(1)
])

# -----------------------------
# 5. Loss and Optimizer
# -----------------------------
optimizer = Adam(learning_rate=0.001, decay=1e-4)
model.compile(optimizer=optimizer, loss='mse')

# -----------------------------
# 6. Training
# -----------------------------
num_epochs = 150
batch_size = 10

history = model.fit(
    X_train, y_train,
    epochs=num_epochs,
    batch_size=batch_size,
    verbose=1
)

# -----------------------------
# 7. Predictions & Performance
# -----------------------------
train_pred = model.predict(X_train)
test_pred  = model.predict(X_test)

train_pred_inv = target_scaler.inverse_transform(train_pred)
test_pred_inv  = target_scaler.inverse_transform(test_pred)

train_r2 = r2_score(target_scaler.inverse_transform(y_train), train_pred_inv)
test_r2  = r2_score(target_scaler.inverse_transform(y_test), test_pred_inv)

print("\nPerformance Results:")
print("Training R2:", train_r2)
print("Testing R2:", test_r2)

# -----------------------------
# 8. Plot training loss
# -----------------------------
plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Training Loss')
plt.legend()
plt.show()