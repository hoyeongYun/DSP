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
from loss.custom_loss import weighted_mse_loss
from model.gpt2_mlp import GPT_MLP_MODEL

# fix seed
np.set_printoptions(linewidth=np.inf)
random_seed = 21
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
# random.seed(random_seed)

# args
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MONTHS = 87
STORES = 642
ITEM_STORES = 11556
in_features = 14
out_features = 1
inter_dim = 56
window_size = 15
n_heads = 4
n_layers = 4
split_frac = 0.8
lr = 1e-4
epochs = 25
batch_size = 48
# patience = 7

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

x_1 = np.load('/workspace/DSP/data/INPUT/window_15_input_x_1.npy')
x_2 = np.load('/workspace/DSP/data/INPUT/window_15_input_x_2.npy')
y = np.load('/workspace/DSP/data/INPUT/window_15_input_y.npy')
x = np.concatenate([x_1, x_2])

# data split
train_size = int(len(x) * split_frac)
train_x = x[:train_size]
train_y = y[:train_size]

eval_x = x[train_size:]
eval_y = y[train_size:]

test_x = data_df[FEAT_KEY].to_numpy().reshape(ITEM_STORES, MONTHS, len(FEAT_KEY))[:, -window_size:, :]

# numpy to tensor
trainX_tensor = torch.FloatTensor(train_x).to(device)
trainY_tensor = torch.FloatTensor(train_y).to(device)

evalX_tensor = torch.FloatTensor(eval_x).to(device)
evalY_tensor = torch.FloatTensor(eval_y).to(device)

testX_tensor = torch.FloatTensor(test_x).to(device)

# shuffle
train_index = torch.randperm(trainX_tensor.size(0)).to(device)
trainX_tensor_sfd = torch.index_select(trainX_tensor, dim=0, index=train_index).to(device)
trainY_tensor_sfd = torch.index_select(trainY_tensor, dim=0, index=train_index).to(device)

eval_index = torch.randperm(evalX_tensor.size(0)).to(device)
evalX_tensor_sfd = torch.index_select(evalX_tensor, dim=0, index=eval_index).to(device)
evalY_tensor_sfd = torch.index_select(evalY_tensor, dim=0, index=eval_index).to(device)

np.save('/workspace/DSP/result/unbiased/gpt/eval_x_index.npy', eval_index.cpu().detach().numpy())

# dataset, dataloader
train_dataset = TensorDataset(trainX_tensor_sfd, trainY_tensor_sfd)
eval_dataset = TensorDataset(evalX_tensor_sfd, evalY_tensor_sfd)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# model
model = GPT_MLP_MODEL(in_features=in_features, 
                    out_features=out_features, 
                    window_size=window_size, 
                    inter_dim=inter_dim, 
                    n_heads=n_heads, 
                    n_layers=n_layers).to(device)

# criterion = nn.MSELoss().to(device)
criterion = weighted_mse_loss
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train_hist = np.zeros(epochs)
train_losses = []
eval_losses = []
count = 0

# train, eval
for epoch in tqdm(range(epochs)):

    model.train()
    for batch_idx, samples in enumerate(train_loader):
        x_train, y_train = samples
        outputs = model(x_train)
        loss = criterion(outputs, y_train, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
    model.eval()
    for x_eval, y_eval in eval_loader:
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

# output result
model.eval()
with torch.no_grad():
    eval_result = model(evalX_tensor).detach().cpu().numpy()
    test_result = model(testX_tensor).detach().cpu().numpy()

    eval_df = pd.DataFrame(eval_result.reshape(-1, 1))
    print(eval_df.shape)
    test_df = pd.DataFrame(test_result.reshape(-1, 1))

    eval_df.to_csv(f'/workspace/DSP/result/unbiased/gpt_w15_4_4_56_{epochs}_eval.csv', index=None)
    test_df.to_csv(f'/workspace/DSP/result/unbiased/gpt_w15_4_4_56_{epochs}_test.csv', index=None)
    