import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

import shap


# -----------------------------
# 1. Reproducibility
# -----------------------------
tf.random.set_seed(42)
np.random.seed(42)


# -----------------------------
# 2. Load dataset (same logic)
# -----------------------------
df = pd.read_excel("data1319.xlsx")

y = df["share"].values.reshape(-1, 1)

exclude_cols = ["share", "mo", "year", "model"]
X = df.drop(columns=exclude_cols).values


# -----------------------------
# 3. Scaling
# -----------------------------
input_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_scaled = input_scaler.fit_transform(X)
y_scaled = target_scaler.fit_transform(y)


# -----------------------------
# 4. Train/validation/test split
# -----------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y_scaled, test_size=0.3, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)


# -----------------------------
# 5. Improved model architecture
# -----------------------------
def build_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu',
              kernel_regularizer=regularizers.l2(0.001),
              input_shape=(input_dim,)),
        Dropout(0.2),

        Dense(32, activation='relu',
              kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.2),

        Dense(16, activation='relu'),

        Dense(1, activation='linear')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return model


model = build_model(X_train.shape[1])

print("\nModel Summary:")
model.summary()


# -----------------------------
# 6. Early stopping
# -----------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)


# -----------------------------
# 7. Training
# -----------------------------
history = model.fit(
    X_train, y_train,
    epochs=300,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)


# -----------------------------
# 8. Predictions (IMPORTANT: inverse scaling)
# -----------------------------
train_pred_scaled = model.predict(X_train)
test_pred_scaled = model.predict(X_test)

train_pred = target_scaler.inverse_transform(train_pred_scaled)
test_pred = target_scaler.inverse_transform(test_pred_scaled)

y_train_orig = target_scaler.inverse_transform(y_train)
y_test_orig = target_scaler.inverse_transform(y_test)


# -----------------------------
# 9. Evaluation
# -----------------------------
def evaluate(y_true, y_pred, label=""):
    print(f"\n=== {label} ===")
    print("R2:", r2_score(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("MAE:", mean_absolute_error(y_true, y_pred))


evaluate(y_train_orig, train_pred, "Train")
evaluate(y_test_orig, test_pred, "Test")


# -----------------------------
# 10. Plot learning curves
# -----------------------------
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title('Training vs Validation Loss')
plt.show()


# -----------------------------
# 11. K-Fold Cross Validation (manual)
# -----------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = []

for train_idx, test_idx in kf.split(X_scaled):
    X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
    y_tr, y_te = y_scaled[train_idx], y_scaled[test_idx]

    m = build_model(X_tr.shape[1])

    m.fit(
        X_tr, y_tr,
        epochs=150,
        batch_size=16,
        verbose=0
    )

    pred = m.predict(X_te)
    pred = target_scaler.inverse_transform(pred)
    y_te_orig = target_scaler.inverse_transform(y_te)

    cv_r2.append(r2_score(y_te_orig, pred))

print("\n=== Cross-validation ===")
print("R2 scores:", cv_r2)
print("Mean R2:", np.mean(cv_r2))


# -----------------------------
# 12. SHAP Explainability
# -----------------------------
explainer = shap.KernelExplainer(
    model.predict,
    X_scaled[:100]  # small background sample
)

shap_values = explainer.shap_values(X_scaled[:200])

shap.summary_plot(shap_values, X_scaled[:200])