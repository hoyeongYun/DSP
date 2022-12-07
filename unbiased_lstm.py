import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.utils.data import DataLoader, TensorDataset

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

data_df = data_df[['Item_Store_Key', 'Month', 'Store', 'Urban_Rural', 'Location_Cluster', 'Item_Type', 'Item', 'Industry_Size', 
                    'Retail_Size', 'Target_3_Month_Retail_Sum', 'Sales_At_The_Month_Total', 'Sales_At_The_Month_Per_Item', 'Sales_At_The_Month_Per_Item_Type']]
item_ord = ['Power Cord', 'Phone Charger', 'Ear Buds','Mouse', 'Keyboard', 'Milk','Eggs', 'Cereal', 
            'Shrimp','Noodles', 'Steak', 'King Crab','Tape', 'Glue', 'Nails','Bracket', 'Brush', 'Paint']
type_ord = ['Electronics', 'Grocery', 'Home Goods']
item_ord_d = CategoricalDtype(categories = item_ord, ordered = True) 
type_ord_d = CategoricalDtype(categories = type_ord, ordered = True) 
data_df['Item'] = data_df['Item'].astype(item_ord_d)
data_df['Item_Type'] = data_df['Item_Type'].astype(type_ord_d)

data_df = pd.get_dummies(data_df, columns = ['Urban_Rural', 'Item_Type', 'Location_Cluster'])

TARGET_KEY = ['Target_3_Month_Retail_Sum']
FEAT_KEY = ['Industry_Size', 'Retail_Size', 
            'Sales_At_The_Month_Total', 'Sales_At_The_Month_Per_Item', 'Sales_At_The_Month_Per_Item_Type',  
            'Location_Cluster_0', 'Location_Cluster_1', 'Location_Cluster_2', 'Location_Cluster_3',
            'Item_Type_Electronics', 'Item_Type_Grocery', 'Item_Type_Home Goods', 
              'Urban_Rural_Rural', 'Urban_Rural_Urban']
window_size = 15

# x = np.empty((0, window_size, len(FEAT_KEY)), dtype=np.float32)
# y = np.empty((0, len(TARGET_KEY)), dtype=np.float32)

# for store_id in tqdm(range(1, 643)):
#   cur_store_df = data_df[data_df.Store == store_id]
#   nonzero_index = cur_store_df[(cur_store_df.Month >= (window_size - 1)) & (cur_store_df.Target_3_Month_Retail_Sum != 0)].index
#   for i in nonzero_index:
#     x = np.append(x, data_df[FEAT_KEY].iloc[i-(window_size-1) : i+1].to_numpy().reshape(1, window_size, len(FEAT_KEY)), axis=0)
#     y = np.append(y, data_df[TARGET_KEY].iloc[i].to_numpy().reshape(1, len(TARGET_KEY)), axis=0)

# test_x = data_df[FEAT_KEY].to_numpy().reshape(11556, 87, len(FEAT_KEY))[:, -window_size:, :]

# np.save('/workspace/DSP/INPUT/data_w15_input_x', x)
# np.save('/workspace/DSP/INPUT/data_w15_input_y', y)

x_1 = np.load('/workspace/DSP/data/INPUT/window_15_input_x_1.npy')
x_2 = np.load('/workspace/DSP/data/INPUT/window_15_input_x_2.npy')
y = np.load('/workspace/DSP/data/INPUT/window_15_input_y.npy')
x = np.concatenate([x_1, x_2])
test_x = data_df[FEAT_KEY].to_numpy().reshape(11556, 87, len(FEAT_KEY))[:, -window_size:, :]

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, output_dim, layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.layers = layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, (hidden_dim)//2),
                                nn.ReLU(),
                                nn.Linear((hidden_dim)//2, output_dim))
        # self.mlp1 = nn.Sequential(
        #     nn.Linear(hidden_dim, (hidden_dim)//2),
        #     nn.ReLU(),
        #     nn.Linear((hidden_dim)//2, output_dim)
        # )

        # self.mlp2 = nn.Sequential(
        #     nn.Linear(seq_len, (seq_len)//2),
        #     nn.ReLU(),
        #     nn.Linear((seq_len)//2, output_dim)
        # )

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.layers, self.seq_len, self.hidden_dim),
            torch.zeros(self.layers, self.seq_len, self.hidden_dim)
        )
    def forward(self, x):
        x, _status = self.lstm(x)
        x = self.fc(x[:, -1])
        # x = self.mlp1(x)
        # x = self.mlp2(x[:, :, 0])
        return x

def weighted_mse_loss(result, target, device):
    w_zero = 0.05
    max_target_val = 145.0
    weight = torch.FloatTensor(np.where(target.cpu().detach().numpy() < 1.0, w_zero, np.exp(target.cpu().detach().numpy() / max_target_val))).to(device)
    return torch.sum(weight * (result - target) ** 2)

# 일단 scaling 안하고

train_size = int(len(x) * 0.9)
train_x = x[:train_size]
train_y = y[:train_size]

eval_x = x[train_size:]
eval_y = y[train_size:]

trainX_tensor = torch.FloatTensor(train_x).to(device)
trainY_tensor = torch.FloatTensor(train_y).to(device)

evalX_tensor = torch.FloatTensor(eval_x).to(device)
evalY_tensor = torch.FloatTensor(eval_y).to(device)

testX_tensor = torch.FloatTensor(test_x).to(device)

train_dataset = TensorDataset(trainX_tensor, trainY_tensor)
eval_dataset = TensorDataset(evalX_tensor, evalY_tensor)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False, drop_last=True)

model = LSTM(input_dim=14, hidden_dim=128, seq_len=window_size, output_dim=1, layers=2).to(device)

lr = 1e-3
epochs = 200
# criterion = nn.MSELoss().to(device)
criterion = weighted_mse_loss
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train_hist = np.zeros(epochs)

train_losses = []
eval_losses = []
count = 0
patience = 20

for epoch in tqdm(range(epochs)):

    model.train()
    for batch_idx, samples in enumerate(train_loader):
        x_train, y_train = samples
        model.reset_hidden_state()
        outputs = model(x_train)
        loss = criterion(outputs, y_train, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
    model.eval()
    for x_eval, y_eval in eval_loader:
        model.reset_hidden_state()
        outputs = model(x_eval)
        loss = criterion(outputs, y_eval, device)
        eval_losses.append(loss.item())
    
    train_loss = np.average(train_losses)
    eval_loss = np.average(eval_losses)
    train_hist[epoch] = eval_loss

    epoch_len = len(str(epoch))
    print('\n')
    print(f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                        f'train_loss: {train_loss:.5f} ' +
                        f'valid_loss: {eval_loss:.5f}')
  
    if train_hist[epoch-1] < train_hist[epoch]:
        count += 1
        print(count)
    if count >= patience:
        print('\nEarly Stopping')
        break

model.eval()
with torch.no_grad():
    model.reset_hidden_state()
    eval_result = model(evalX_tensor).detach().cpu().numpy()
    model.reset_hidden_state()
    test_result = model(testX_tensor).detach().cpu().numpy()

    eval_df = pd.DataFrame(eval_result.reshape(-1, 1))
    test_df = pd.DataFrame(test_result.reshape(-1, 1))

    eval_df.to_csv(f'/workspace/DSP/result/unbiased/w15_eval.csv', index=None)
    test_df.to_csv(f'/workspace/DSP/result/unbiased/w15_test.csv', index=None)
    