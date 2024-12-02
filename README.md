# LSTM_Pollution_Forecasting

This project focuses on forecasting pollution levels (PM2.5) using an LSTM (Long Short-Term Memory) model, based on multivariate environmental data. The goal is to predict future pollution levels by training the model on historical data, including temperature, humidity, pressure, wind speed, snow, and rain.

![air_pollution](https://github.com/user-attachments/assets/aba07557-389c-4b05-b2a8-ff815f79fa6d)

## Project Overview

- **Objective:** Predict PM2.5 levels using LSTM for multivariate time series forecasting.
- **Data:** The dataset includes features such as pollution, dew point, temperature, pressure, wind speed, snow, and rain.
- **Model:** LSTM (Long Short-Term Memory) model with one hidden layer and a dense output layer for regression.
- **Libraries Used:** 
  - `NumPy`, `Pandas` for data manipulation
  - `scikit-learn` for MinMax scaling
  - `TensorFlow/Keras` for building and training the LSTM model
  - `Matplotlib` for data visualization

**Data Preparation**

The data is loaded from a CSV file (LSTM-Multivariate_pollution.csv), which contains historical pollution and weather-related data. MinMaxScaler is applied to scale the features of the dataset to a range between 0 and 1, improving model performance. The data is transformed into sequences with a sliding window of size 24. Each sequence consists of the previous 24 time steps (features) to predict the next pollution value.

**Model Architecture**

The LSTM layer is used with 64 units to process the sequence of environmental features. A Dense layer with 32 neurons and ReLU activation function is used to process the output of the LSTM layer. A Dense layer with a single output unit is used to predict the PM2.5 value.

**Training**

Adam optimizer with a learning rate of 0.001 is used to minimize the loss function. Mean Squared Error (MSE) is used as the loss function to evaluate the model. Mean Absolute Error (MAE) is chosen as a metric for model evaluation. The model is trained for 20 epochs with a batch size of 32.
