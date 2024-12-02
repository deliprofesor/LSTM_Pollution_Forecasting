# LSTM_Pollution_Forecasting

## Description

This project implements a Long Short-Term Memory (LSTM) neural network model to forecast pollution levels (specifically PM2.5) based on multiple environmental variables such as temperature, humidity, pressure, wind speed, and more. The model is trained on historical pollution data and uses a multivariate time series approach to predict future pollution levels.

## Project Overview

- **Objective:** Predict PM2.5 levels using LSTM for multivariate time series forecasting.
- **Data:** The dataset includes features such as pollution, dew point, temperature, pressure, wind speed, snow, and rain.
- **Model:** LSTM (Long Short-Term Memory) model with one hidden layer and a dense output layer for regression.
- **Libraries Used:** 
  - `NumPy`, `Pandas` for data manipulation
  - `scikit-learn` for MinMax scaling
  - `TensorFlow/Keras` for building and training the LSTM model
  - `Matplotlib` for data visualization

## Installation

To run this project, ensure you have the following Python libraries installed:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib
