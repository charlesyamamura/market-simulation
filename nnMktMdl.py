import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Set random seed for reproducibility
tf.random.set_seed(42)

# Load the input and target data
X = pd.read_excel('data1319.xlsx', sheet_name=0, usecols='D:S', skiprows=1, nrows=765).values
y = pd.read_excel('data1319.xlsx', sheet_name=0, usecols='T', skiprows=1, nrows=765).values

# Expand dimensions of target data if necessary
t = np.expand_dims(y, axis=1) if y.ndim == 1 else y

# Normalize input and target data
input_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

x_scaled = input_scaler.fit_transform(X)
t_scaled = target_scaler.fit_transform(t)

# Split data into training, validation, and testing sets
x_train, x_temp, t_train, t_temp = train_test_split(x_scaled, t_scaled, test_size=0.3, random_state=42)
x_val, x_test, t_val, t_test = train_test_split(x_temp, t_temp, test_size=0.5, random_state=42)

# Create a fitting network
hidden_layer_size = 10
model = Sequential([
    Dense(hidden_layer_size, activation='relu', input_shape=(x_scaled.shape[1],)),
    Dense(t_scaled.shape[1], activation='relu')
])

# Display the model summary
print("Model Summary:")
model.summary()

# Compile the model
model.compile(optimizer=Adam(), loss=MeanSquaredError(), metrics=['mae'])

# Train the network
history = model.fit(x_train, t_train, epochs=100, batch_size=10,
                    validation_data=(x_val, t_val), verbose=1)

# Test the network
y_pred = model.predict(x_test)
performance = model.evaluate(x_test, t_test, verbose=0)

# Save the trained model
model_save_path = "mktRegMdl.keras"
model.save(model_save_path)
print(f"Model saved at: {model_save_path}")

# Calculate training and testing adjusted accuracies
train_mse = model.evaluate(x_train, t_train, verbose=0)[0]
test_mse = performance[0]
target_variance = np.var(t_scaled)

train_accuracy = max(0, 1 - (train_mse / target_variance))
test_accuracy = max(0, 1 - (test_mse / target_variance))

# Calculate R-squared for training and testing
train_pred = model.predict(x_train)
test_pred = model.predict(x_test)

train_r2 = r2_score(t_train, train_pred)
test_r2 = r2_score(t_test, test_pred)

# Print performance results
print("\nPerformance Results:")
print("Test Performance (MSE):", test_mse)
print("Adjusted Training Accuracy:", train_accuracy)
print("Adjusted Testing Accuracy:", test_accuracy)
print("Training R-squared:", train_r2)
print("Testing R-squared:", test_r2)

# Plot training and validation loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss')
plt.show()

# Plot adjusted accuracy results
epochs = range(1, len(history.history['loss']) + 1)
train_accuracy_epoch = [max(0, 1 - (loss / target_variance)) for loss in history.history['loss']]
val_accuracy_epoch = [max(0, 1 - (loss / target_variance)) for loss in history.history['val_loss']]
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_accuracy_epoch, label='Training Adjusted Accuracy')
plt.plot(epochs, val_accuracy_epoch, label='Validation Adjusted Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Adjusted Accuracy')
plt.title('Training and Validation Adjusted Accuracy')
plt.show()

# Optional: Generate a simple function for deployment
def my_neural_network_function(new_x):
    new_x_scaled = input_scaler.transform(new_x)
    y_scaled = model.predict(new_x_scaled)
    return target_scaler.inverse_transform(y_scaled)
