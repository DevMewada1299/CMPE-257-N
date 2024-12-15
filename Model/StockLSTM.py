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

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):

        lstm_out, _ = self.lstm(x)

        last_hidden_state = lstm_out[:, -1, :]
        output = self.fc(last_hidden_state)
        return output


def createModel():
    input_size = 6  
    hidden_size = 64
    num_layers = 2
    output_size = 3 

    model = StockLSTM(input_size, hidden_size, num_layers, output_size)

    return model