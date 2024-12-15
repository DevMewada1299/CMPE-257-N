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
from  data.data_download import createDataLoader
from Model.StockLSTM import createModel
import sys
import os

# Add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


dataloader = createDataLoader()
model = createModel()


def trainModel():

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    epochs = 200
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(X_batch)
            
            # Compute loss
            loss = criterion(predictions, y_batch)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

trainModel()

torch.save(model.state_dict(), 'saved_model/stock_price_model.pth')