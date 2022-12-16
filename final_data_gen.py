import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.utils.data import DataLoader, TensorDataset
from pandas.api.types import CategoricalDtype

torch.manual_seed(0)
np.set_printoptions(linewidth=np.inf)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_df = pd.read_csv('/workspace/DSP/data/PREPROCESSED/Unbiased_Data.csv')

data_df.Month = data_df.Month.astype(np.int16)
data_df.Store = data_df.Store.astype(np.int16)
data_df.Store_Owner = data_df.Store_Owner.astype(np.int16)
data_df.Latitude = data_df.Latitude.astype(np.float32)
data_df.Longitude = data_df.Longitude.astype(np.float32)
data_df.Industry_Size = data_df.Industry_Size.astype(np.int16)
data_df.Retail_Size = data_df.Retail_Size.astype(np.int16)
data_df.Target_3_Month_Retail_Sum = data_df.Target_3_Month_Retail_Sum.astype(np.float32)
data_df.Item_Store_Key = data_df.Item_Store_Key.astype(np.int16)
data_df.Sales_At_The_Month_Total = data_df.Sales_At_The_Month_Total.astype(np.float32)
data_df.Sales_At_The_Month_Per_Item = data_df.Sales_At_The_Month_Per_Item.astype(np.float32)
data_df.Sales_At_The_Month_Per_Item_Type = data_df.Sales_At_The_Month_Per_Item_Type.astype(np.float32)

data_df = data_df[['Item_Store_Key', 'Month', 'Store', 'Store_Owner', 'Urban_Rural', 'Latitude', 'Longitude', 'Item_Type', 'Item', 'Industry_Size', 
                    'Retail_Size', 'Target_3_Month_Retail_Sum', 'Sales_At_The_Month_Total', 'Sales_At_The_Month_Per_Item', 'Sales_At_The_Month_Per_Item_Type']]

item_ord = ['Power Cord', 'Phone Charger', 'Ear Buds','Mouse', 'Keyboard', 'Milk','Eggs', 'Cereal', 'Shrimp','Noodles', 'Steak', 'King Crab','Tape', 'Glue', 'Nails','Bracket', 'Brush', 'Paint']
type_ord = ['Electronics', 'Grocery', 'Home Goods']
item_ord_d = CategoricalDtype(categories = item_ord, ordered = True) 
type_ord_d = CategoricalDtype(categories = type_ord, ordered = True) 
data_df['Item'] = data_df['Item'].astype(item_ord_d)
data_df['Item_Type'] = data_df['Item_Type'].astype(type_ord_d)

lati_scaler = MinMaxScaler()
long_scaler = MinMaxScaler()
industry_scaler = MinMaxScaler()

data_df.Latitude = lati_scaler.fit_transform(data_df.Latitude.to_numpy().reshape(-1, 1))
data_df.Longitude = long_scaler.fit_transform(data_df.Longitude.to_numpy().reshape(-1, 1))
data_df.Industry_Size = industry_scaler.fit_transform(data_df.Industry_Size.to_numpy().reshape(-1, 1))

data_df = pd.get_dummies(data_df, columns = ['Urban_Rural'])

FEAT_KEY = ['Sales_At_The_Month_Total', 
            'Sales_At_The_Month_Per_Item', 
            'Sales_At_The_Month_Per_Item_Type',
            'Latitude', 'Longitude',
            'Urban_Rural_Rural', 'Urban_Rural_Urban',
            'Industry_Size', 
            'Retail_Size']
ADD_KEY = ['Store', 'Item', 'Target_3_Month_Retail_Sum', 'Month']

window_size = 18
x = np.empty((0, window_size, len(FEAT_KEY)), dtype=np.float16)
y = np.empty((0, window_size, 3), dtype=np.float16)

for store_id in tqdm(range(1, 643)):
  for item_name in ['Power Cord', 'Phone Charger', 'Ear Buds','Mouse', 'Keyboard', 'Milk','Eggs', 'Cereal', 'Shrimp','Noodles', 'Steak', 'King Crab','Tape', 'Glue', 'Nails','Bracket', 'Brush', 'Paint']:
    cur_store_item_df = data_df[(data_df.Store == store_id) & (data_df.Item == item_name)][ADD_KEY + FEAT_KEY].reset_index(drop=True)
    nonzero_index = cur_store_item_df[(cur_store_item_df.Month >= (window_size - 1)) & (cur_store_item_df.Target_3_Month_Retail_Sum != 0)].index
    cur_data = cur_store_item_df.to_numpy()

    for i in nonzero_index:
      x = np.append(x, cur_data[i-(window_size-1):i+1, 4:].reshape(1, window_size, len(FEAT_KEY)), axis=0)
      y_temp = np.empty((18, 3), dtype=np.float16)
      for d in range(1, 4):
        y_temp[:, d-1] = cur_data[i-(window_size-1)+d : i+1+d, -1]
      y = np.append(y, y_temp.reshape(1, window_size, 3), axis=0)

print('x.shape : ', x.shape)
print('y.shape : ', y.shape) 

np.save(f'/workspace/DSP/data/INPUT/final/final_input_{len(x)}_18_9_x', x)
np.save(f'/workspace/DSP/data/INPUT/final/final_input_{len(y)}_18_3_y', y)
