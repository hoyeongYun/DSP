import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
# import cupy as cp
from pandas.api.types import CategoricalDtype

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.utils.data import DataLoader, TensorDataset

def is_zero_data(total_retail):
  if total_retail == 0:
    return 0
  else:
    return 1

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

data_df['Has_Retail'] = data_df.Target_3_Month_Retail_Sum.apply(is_zero_data)

data_df = data_df[['Item_Store_Key', 'Month', 'Store', 'Urban_Rural', 'Location_Cluster', 'Item_Type', 'Item', 'Industry_Size',
                    'Retail_Size', 'Target_3_Month_Retail_Sum', 'Sales_At_The_Month_Total', 'Sales_At_The_Month_Per_Item', 'Sales_At_The_Month_Per_Item_Type', 'Has_Retail']]

item_ord = ['Power Cord', 'Phone Charger', 'Ear Buds','Mouse', 'Keyboard', 'Milk','Eggs', 'Cereal', 'Shrimp','Noodles', 'Steak', 'King Crab','Tape', 'Glue', 'Nails','Bracket', 'Brush', 'Paint']
type_ord = ['Electronics', 'Grocery', 'Home Goods']
item_ord_d = CategoricalDtype(categories = item_ord, ordered = True)
type_ord_d = CategoricalDtype(categories = type_ord, ordered = True)
data_df['Item'] = data_df['Item'].astype(item_ord_d)
data_df['Item_Type'] = data_df['Item_Type'].astype(type_ord_d)

data_df = pd.get_dummies(data_df, columns = ['Urban_Rural', 'Item_Type', 'Location_Cluster'])


# non zero data gen
TARGET_KEY = ['Has_Retail']
FEAT_KEY = ['Industry_Size', 'Retail_Size',
            'Sales_At_The_Month_Total', 'Sales_At_The_Month_Per_Item', 'Sales_At_The_Month_Per_Item_Type',
            'Location_Cluster_0', 'Location_Cluster_1', 'Location_Cluster_2', 'Location_Cluster_3',
            'Item_Type_Electronics', 'Item_Type_Grocery', 'Item_Type_Home Goods',
              'Urban_Rural_Rural', 'Urban_Rural_Urban']
window_size = 15

x = np.empty((0, window_size, len(FEAT_KEY)), dtype=np.float16)

for store_id in tqdm(range(1, 643)):
  cur_store_df = data_df[data_df.Store == store_id]
  nonzero_index = cur_store_df[(cur_store_df.Month >= (window_size - 1)) & (cur_store_df.Target_3_Month_Retail_Sum != 0)].index
  for i in nonzero_index:
    x = np.append(x, data_df[FEAT_KEY].iloc[i-(window_size-1) : i+1].to_numpy().reshape(1, window_size, len(FEAT_KEY)), axis=0)

y = np.ones((x.shape[0], 1))

x_1 = x[:len(x) // 2]
x_2 = x[len(x) // 2:]

np.save('/workspace/DSP/data/INPUT/clsf/zero_clsf_w15_input_x_1', x_1)
np.save('/workspace/DSP/data/INPUT/clsf/zero_clsf_w15_input_x_2', x_2)
np.save('/workspace/DSP/data/INPUT/clsf/zero_clsf_w15_input_y', y)


# zero data gen

# four_to_six_idx = [15, 16, 17, 27, 28, 29, 39, 40, 41, 51, 52, 53, 63, 64, 65, 75, 76, 77]

x_zero = np.empty((0, window_size, len(FEAT_KEY)), dtype=np.float16)

for store_id in tqdm(range(1, 643)):
  cur_store_df = data_df[data_df.Store == store_id].reset_index(drop=True)
  temp_df = cur_store_df[(cur_store_df.Month >= (window_size - 1)) & (cur_store_df.Target_3_Month_Retail_Sum == 0)]
  nonzero_index = temp_df[(temp_df.Month == 15) | (temp_df.Month == 16) | (temp_df.Month == 17) |
                          (temp_df.Month == 27) | (temp_df.Month == 28) | (temp_df.Month == 29) |
                          (temp_df.Month == 39) | (temp_df.Month == 40) | (temp_df.Month == 41) |
                          (temp_df.Month == 51) | (temp_df.Month == 52) | (temp_df.Month == 53) |
                          (temp_df.Month == 63) | (temp_df.Month == 64) | (temp_df.Month == 65) |
                          (temp_df.Month == 75) | (temp_df.Month == 76) | (temp_df.Month == 77) ].index
  for i in nonzero_index:
    x_zero = np.append(x_zero, cur_store_df[FEAT_KEY].iloc[i-(window_size-1) : i+1].to_numpy().reshape(1, window_size, len(FEAT_KEY)), axis=0)

y_zero = np.zeros((x_zero.shape[0], 1))

x_zero_1 = x_zero[:len(x_zero) // 2]
x_zero_2 = x_zero[len(x_zero) // 2:]

np.save('/workspace/DSP/data/INPUT/clsf/zero_clsf_w15_input_x_zero_456_1', x_zero_1)
np.save('/workspace/DSP/data/INPUT/clsf/zero_clsf_w15_input_x_zero_456_2', x_zero_2)
np.save('/workspace/DSP/data/INPUT/clsf/zero_clsf_w15_input_y_zero_456', y_zero)
