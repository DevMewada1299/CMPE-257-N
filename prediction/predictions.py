import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import yfinance as yf
import matplotlib.pyplot as plt
import torch.nn
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from Model.StockLSTM import StockLSTM
from data.data_download import saved_df
from data.data_download import scaler_dict
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error

input_size = 6  
hidden_size = 64
num_layers = 2
output_size = 3 

model = StockLSTM(input_size, hidden_size, num_layers, output_size)
model.load_state_dict(torch.load('saved_model/stock_price_model.pth'))
scaled_data = saved_df[1]
sequence_length = 10 
scaler = scaler_dict[0]


model.eval()
with torch.no_grad():
    
    input_sequence = scaled_data[-sequence_length:, :]
    input_sequence = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Predict
    predictions = model(input_sequence)
    predictions = predictions.numpy()

    # Denormalize predictions
    def denormalize(scaler, data, col_indices):
        placeholder = np.zeros((data.shape[0], scaler.n_features_in_))
        placeholder[:, col_indices] = data
        return scaler.inverse_transform(placeholder)[:, col_indices]

    col_indices = [2, 3, 1]  # High, Low, Close columns
    denorm_predictions = denormalize(scaler, predictions, col_indices)

    print(input_sequence.shape,predictions.shape)

    #print("RMSE :", root_mean_squared_error(input_sequence.numpy(), predictions))
    # print("RMSE :", mean_absolute_error(y_test.numpy()[:], test_predictions[:]))
    # print("RMSE :", mean_squared_error(y_test.numpy()[:], test_predictions[:]))

    

