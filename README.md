# LSTM_Pollution_Forecasting

This project focuses on forecasting pollution levels (PM2.5) using an LSTM (Long Short-Term Memory) model, based on multivariate environmental data. The goal is to predict future pollution levels by training the model on historical data, including temperature, humidity, pressure, wind speed, snow, and rain.

![air_pollution](https://github.com/user-attachments/assets/aba07557-389c-4b05-b2a8-ff815f79fa6d)

## Required Libraries

At the beginning of the code, the necessary libraries for data processing and modeling are imported:

- **NumPy** and **Pandas** for data handling,
- **MinMaxScaler** for data scaling,
- **LSTM**, **Dense**, and Adam for **Keras** model and optimization,
- **Matplotlib** for visualization.

## Loading the Data

The data is loaded from the LSTM-Multivariate_pollution.csv file, which contains various environmental factors related to pollution and weather. After loading the data, the column names and the first 5 rows are printed to get an overview of the dataset.

## Data Preparation

- **Feature Selection:** Environmental factors like pollution, dew, temp, press, wnd_spd, snow, and rain are selected as the features.
- **Data Normalization:** The features are scaled between 0 and 1 using MinMaxScaler, which helps the model perform better.
  
## Creating Sequences

- **Lag Features (Sequence Creation):** For time series data, lag features are created by using the previous time steps (24 time steps) to predict the next pollution value.
  
## Splitting the Data

The data is split into training and test datasets, with 80% used for training and 20% for testing. The training data is used to train the model, and the test data is used to evaluate its performance.

## Model Architecture

The model is built using LSTM and Dense layers:

- **LSTM Layer:** This layer has 64 units and processes the time series data.
- **Dense Layer:** This layer has 32 neurons and processes the data from the LSTM layer.
- **Output Layer:** A single neuron is used in the output layer to predict the PM2.5 value.
  
## Model Training

The Adam optimizer is used with the Mean Squared Error (MSE) loss function. The model is trained for 20 epochs with a batch size of 32. During training, validation data is used to track the model’s performance.

## Model Evaluation and Visualization

Predictions are made on the test data. The normalized data is scaled back to the original range. The first 100 predicted and actual PM2.5 values are plotted for comparison. In the plot, the blue line represents the actual PM2.5 values, and the red line represents the predicted PM2.5 values by the model. This graph visually helps to analyze how well the model’s predictions align with the actual data.

![PM2 5](https://github.com/user-attachments/assets/9f90d2e3-2bd2-4b29-ad68-4f1f6e0a36be)

