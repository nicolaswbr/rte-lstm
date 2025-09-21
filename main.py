from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)

data = pd.read_csv('eCO2mix_RTE_En-cours-TR.csv', delimiter=';')

# Drop rows with missing Date or Consommation
data = data.dropna(subset=['Date', 'Consommation'])

# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Drop rows where 'Date' could not be converted
data = data.dropna(subset=['Date'])

print(data.head())
print(data.info())
print(data.describe())

# Initial Data Visualization
# Plot 1 -  Time Series of a Key Variable

plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Consommation'], label="Consommation", color='blue')
plt.title('Consommation Over Time')
plt.legend()
plt.show()


# Prepare data for LSTM 
consumption = data['Consommation']
dataset = consumption.values
training_data_len = int(np.ceil(len(dataset) * 0.95)) 

# Preprocessing Stages

scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset.reshape(-1, 1))

training_data = scaled_data[:training_data_len]

X_train, Y_train = [], []

# create a sliding window for consumption data 
for i in range(60, len(training_data)):
    X_train.append(training_data[i-60:i, 0])
    Y_train.append(training_data[i, 0])

X_train, Y_train = np.array(X_train), np.array(Y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build the LSTM model
model = keras.Sequential()
model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(keras.layers.LSTM(64, return_sequences=False))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(1))

model.summary()
model.compile(optimizer='adam', loss='mae', metrics=[keras.metrics.MeanAbsoluteError()])

training = model.fit(X_train, Y_train, epochs=1, batch_size=32)
model.save('rte_lstm_model.h5')

test_data = scaled_data[training_data_len - 60:]
X_test = []
Y_test = dataset[training_data_len:]

for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Plotting the results
train = data[:training_data_len]
test = data[training_data_len:]

test = test.copy()
test['Predictions'] = predictions

plt.figure(figsize=(16, 8))
plt.plot(train['Date'], train['Consommation'], label='Training Data', color='blue')
plt.plot(test['Date'], test['Consommation'], label='Actual Consommation', color='green')
plt.plot(test['Date'], test['Predictions'], label='Predicted Consommation', color='red')
plt.title('Consommation Prediction')
plt.xlabel('Date')
plt.ylabel('Consommation')
plt.legend()
plt.show()  


