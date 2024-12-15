import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import yfinance as yf
import matplotlib.pyplot as plt
import torch.nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader




class StockDataset(Dataset):
    def __init__(self, data, sequence_length=10, target_days=5):
        """
        Args:
        - data: Scaled dataset as a NumPy array.
        - sequence_length: Number of past days used as input.
        - target_days: Number of future days for which we predict High, Low, Avg_Close.
        """
        self.data = data
        self.sequence_length = sequence_length
        self.target_days = target_days

    def __len__(self):
        return len(self.data) - self.sequence_length - self.target_days + 1

    def __getitem__(self, idx):
        # Extract input sequence
        X = self.data[idx:idx + self.sequence_length, :]
        
        # Extract future data for target calculation
        future_data = self.data[idx + self.sequence_length:idx + self.sequence_length + self.target_days]
        
        # Calculate target values
        high = future_data[:, 2].max()  # High price column
        low = future_data[:, 3].min()   # Low price column
        avg_close = future_data[:, 1].mean()  # Close price column

        # Return tensors
        y = np.array([high, low, avg_close], dtype=np.float32)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
def getDataFrameForTraining():
    
    df = yf.download('NVDA', start='2024-01-01', end='2024-12-14')

    return df

def createDataLoader():

    scaled_data = scaledData()
    sequence_length = 10  
    target_days = 10     
    dataset = StockDataset(scaled_data, sequence_length, target_days)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    return dataloader

def scaledData():

    df = getDataFrameForTraining()
    scaler = MinMaxScaler()
    scaler_dict[0] = scaler
    scaled_data = scaler.fit_transform(df)

    return scaled_data



saved_df = {}
scaler_dict = {}
saved_df[1] = scaledData()


