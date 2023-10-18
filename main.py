# library's
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
from datetime import datetime
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# Load Data
company = 'TATAPOWER'
yf.pdr_override()
y_simbols = ['TATAPOWER.NS']
start = datetime(2012, 1, 1)
end = datetime(2020, 1, 1)

data = yf.download(y_simbols, start=start, end=end)

# Prepare Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

predict_days = 60

x_train = []
y_train = []

for x in range(predict_days, len(scaled_data)):
    x_train.append(scaled_data[x - predict_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build The Model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Prediction on the next closing day

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

''' Test The Model Accuracy on Existing Data'''
test_start = datetime(2020, 1, 1)
test_end = datetime.now()

test_data = yf.download(y_simbols, start=test_start, end=test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - predict_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

# Make Predictions on Test Data

x_test = []

for x in range(predict_days, len(model_inputs)):
    x_test.append(model_inputs[x-predict_days:x, 0])


x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot The Test Predictions
plt.plot(actual_prices, color='black', label=f"Actual {company} price")
plt.plot(predicted_prices, color='green', label=f"Predict {company} price")
plt.title(f"{company} Share price")
plt.xlabel('Time')
plt.ylabel(f"{company} Share price")
plt.legend()
plt.show()
