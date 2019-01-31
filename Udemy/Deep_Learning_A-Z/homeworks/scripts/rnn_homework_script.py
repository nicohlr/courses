import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

############################## Data preprocessing ##############################

path_train = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath('__file__')))), 'ressources/Recurrent_Neural_Networks/Google_Stock_Price_Train.csv')
dataset = pd.read_csv(path_train)
dataset.head()

# plot google stock pr
fig = plt.figure(figsize=(20,5))
plt.plot(dataset['Date'], dataset['Open'])
plt.yticks([k*100 for k in range(10)])
plt.xticks([])
plt.show()

# select "Open" column
training_set = dataset.iloc[:, 1:2].values

# Normalise column
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

# creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []

for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

# reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train.shape

############################## Building RNN ##############################

# Initializing the RNN
regressor = Sequential()# Adding the first LSTM layer and some dropout regularization
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
# Adding other LSTM layers
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
# Adding output layer
regressor.add(Dense(units=1))
# Compile RNN
regressor.compile(optimizer='adam', loss='mse')

# Train the RNN
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

############################## Making prediction ##############################

# Getting the real stock price of 1rst month of 2017
path_test = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath('__file__')))), 'ressources/Recurrent_Neural_Networks/Google_Stock_Price_Test.csv')
dataset_test = pd.read_csv(path_test)
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 1rst month of 2017
dataset_total = pd.concat((dataset['Open'], dataset_test['Open']), 0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Make prediction
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
print(rmse)