import pandas as pd
import numpy as np
import torch

class Monthly_Prediction_Dataset(torch.utils.data.Dataset):
    '''
    Args:
        df: input data
            ['Month', 'Store', 'Store_Owner', 'Urban_Rural', 'Latitude', 
            'Longitude', 'Month_Industry_Size', 'Month_Retail_Size', 'Target']
            train_df -> [578*86, 9]
            eval_df -> [64*86, 9]
        preprocs: dictionary of LabelEncoder and MinMaxScaler correspond to each column
        output: x -> [self.len, 86, 7]
                y -> [self.len, 86, 1]

        self.x: [B, 86, 7]
        self.y: [B, 86, 1]
        
        at test mode -> 642, 86, 7
    '''
    def __init__(self, df: pd.DataFrame, device: torch.device, mode: str = 'train_eval'):
        super().__init__()
        self.df = df
        self.mode = mode
        self.total_months = 86
        self.len = int(len(df) // self.total_months)
        self.device = device
        if self.mode == 'train_eval':
            self.x = torch.FloatTensor(df.to_numpy()[:, 1:].reshape(-1, self.total_months, 8)[:, :, :-1]).to(self.device)
            self.y = torch.FloatTensor(df.to_numpy()[:, 1:].reshape(-1, self.total_months, 8)[:, :, -1:]).to(self.device)
        else:
            self.x = torch.FloatTensor(df.to_numpy()[:, 1:].reshape(-1, self.total_months, 7)).to(self.device)
            self.y = None

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return self.len
