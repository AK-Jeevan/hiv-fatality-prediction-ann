# This Model is used to predict HIV/AIDS fatalities using Artificial Neural Networks (ANN)
# Regression Problem

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping

# Step 1: Load and inspect the data
data = pd.read_csv(r"C:\Users\akjee\Documents\AI\DL\ANN\Annual cause death numbers new.csv")
print(data.head())
print(data.describe())

# Step 2: Clean the data
data = data.drop_duplicates()
data = data.dropna().reset_index(drop=True)

# Step 3: Select features (X) and target (y)
# Drop non-numeric columns
data = data.drop(columns=["Entity", "Code", "Year"], errors='ignore')

# Define target and features
target_col = [col for col in data.columns if "HIV" in col][0]
X = data.drop(columns=[target_col])
y = data[target_col]

# Step 4: Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Scale the features using RobustScaler (better for outliers)
scaler = RobustScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Step 6: Build the ANN model (optimized for tabular regression)
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(x_train_scaled.shape[1],)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Output layer for regression

# Step 7: Compile the model
model.compile(optimizer='adam', loss='mae', metrics=['mse'])

# Step 8: Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss', patience=20, restore_best_weights=True, verbose=1, mode='min', min_delta=0.001
)

# Step 9: Train the model
history = model.fit(
    x_train_scaled, y_train,
    epochs=200,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Step 10: Evaluate the model
y_pred = model.predict(x_test_scaled).flatten()

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MAE: {mae:.2f}")
print(f"Test MSE: {mse:.2f}")
print(f"Test R2 Score: {r2:.3f}")

# Step 11: Plot training history
plt.plot(history.history['loss'], label='Train MAE')
plt.plot(history.history['val_loss'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('Loss (MAE)')
plt.legend(loc='upper right')
plt.title('Model Training History')
plt.show()

# Step 12: Plot predictions vs actual
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel('Actual HIV/AIDS Fatalities')
plt.ylabel('Predicted HIV/AIDS Fatalities')
plt.title('Actual vs Predicted HIV/AIDS Fatalities')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()
