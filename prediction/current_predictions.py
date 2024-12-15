import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import yfinance as yf
import matplotlib.pyplot as plt
import torch.nn
from Model.StockLSTM import StockLSTM
from data.data_download import saved_df
from data.data_download import scaler_dict
from datetime import datetime,timedelta
from prediction.predictions import denormalize

input_size = 6  
hidden_size = 64
num_layers = 2
output_size = 3 
sequence_length = 10 
col_indices = [2, 3, 1]

model = StockLSTM(input_size, hidden_size, num_layers, output_size)
model.load_state_dict(torch.load('saved_model/stock_price_model.pth'))
date_input = input("Enter a date (in YYYY-MM-DD format): ")

def getPredsOnCurrentDay():

    try:
        selected_date = datetime.strptime(date_input, "%Y-%m-%d").date()
        print(f"You entered: {selected_date}")
    except ValueError:
        print("Invalid date format! Please enter the date in YYYY-MM-DD format.")

    if selected_date.weekday() == 0:
        s = selected_date - timedelta(days=3)
    else:
        s = selected_date - timedelta(days=1)

    df = yf.download('NVDA', start=s, end=selected_date)

    scaled_data = scaler_dict[0].fit_transform(df)
    scaler = scaler_dict[0]

    new_input_sequence = scaled_data[-sequence_length:, :]  # Use the latest sequence
    new_input_sequence = torch.tensor(new_input_sequence, dtype=torch.float32).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        predictions = model(new_input_sequence).numpy()

    # Denormalize predictions if scaling was applied
    denorm_predictions = denormalize(scaler, predictions, col_indices)

    # Print predicted values
    high, low, avg_close = denorm_predictions[0]
    print(f"Predicted High Price for next 5 days: {high}")
    print(f"Predicted Low Price for next 5 days: {low}")
    print(f"Predicted Avg Closing Price for next 5 days: {avg_close}")

    predicted_values = {}
    predicted_values['high'] = high
    predicted_values['low'] = low
    predicted_values['avg_close'] = avg_close

    return predicted_values


p = getPredsOnCurrentDay() #values returned here

print(p.values())

