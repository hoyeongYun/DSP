from typing import Sequence

import pandas as pd
import numpy as np
import torch

class End_To_End_Dataset(torch.utils.data.Dataset):
    '''
    Args:
        df: input data
            ['Month', 'Store', 'Store_Owner', 'Urban_Rural', 'Latitude', 
            'Longitude', 'Item_Type', 'Item', 'Industry_Size', 'Retail_Size', 'Target_3_Month_Retail_Sum']
            train_df -> [578*84*18, 11]
            eval_df -> [64*84*18, 11]
        preprocs: dictionary of LabelEncoder and MinMaxScaler correspond to each column
        output: x -> [self.len, 86, 8]
                y -> [self.len, 86, 1]
        
        excepted_col = ['Month', 'Store']   'Retail_Size' ablataion study
        self.x: [B, 84, 18D]
        self.y: [B, 84, 18]
        
        at test mode -> 642, 84, 18D
    '''
    def __init__(self, df: pd.DataFrame, 
                device: torch.device, 
                mode: str = 'train_eval', 
                total_months: int = 84,
                total_items: int = 18,
                excepted_col: Sequence[str] = ['Month', 'Store'],
                target_col: Sequence[str] = ['Target_3_Month_Retail_Sum']):
        super().__init__()
        self.df = df
        self.mode = mode
        self.total_months = total_months
        self.total_items = total_items
        self.len = int(len(df) // (self.total_months*self.total_items))
        self.device = device
        self.excepted_col = excepted_col
        self.target_col = target_col
        self.req_cols = len(self.df.columns) - len(self.excepted_col) - len(self.target_col)      
        if self.mode == 'train_eval':
            temp = df.drop(columns=self.excepted_col+self.target_col)
            temp = temp.to_numpy().reshape(-1, self.total_months, self.total_items*self.req_cols)
            self.x = torch.FloatTensor(temp).to(self.device)
            temp = df[self.target_col]
            temp = temp.to_numpy().reshape(-1, self.total_months, self.total_items)
            self.y = torch.FloatTensor(temp).to(self.device)
        else:
            temp = df.drop(columns=self.excepted_col+self.target_col)
            temp = temp.to_numpy().reshape(-1, self.total_months, self.total_items*self.req_cols)
            self.x = torch.FloatTensor(temp).to(self.device)
            self.y = None
        self.store_ids = torch.FloatTensor(df['Store'].to_numpy().reshape(-1, self.total_months*self.total_items)[:, 0]).to(self.device)

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.store_ids[i]

    def __len__(self):
        return self.len
