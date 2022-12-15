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
lr = 3e-3
epochs = 150
batch_size = 48
# patience = 7

data_df = pd.read_csv('/workspace/DSP/data/PREPROCESSED/Unbiased_Data.csv')
data_df = data_df[['Item_Store_Key', 'Month', 'Store', 'Urban_Rural', 'Location_Cluster', 'Item_Type', 'Item', 'Industry_Size', 
                   'Retail_Size', 'Target_3_Month_Retail_Sum', 'Sales_At_The_Month_Total', 'Sales_At_The_Month_Per_Item', 'Sales_At_The_Month_Per_Item_Type']]
data_df = pd.get_dummies(data_df, columns = ['Urban_Rural', 'Item_Type', 'Location_Cluster'])
FEAT_KEY = ['Industry_Size', 'Retail_Size', 
            'Sales_At_The_Month_Total', 'Sales_At_The_Month_Per_Item', 'Sales_At_The_Month_Per_Item_Type',  
            'Location_Cluster_0', 'Location_Cluster_1', 'Location_Cluster_2', 'Location_Cluster_3',
            'Item_Type_Electronics', 'Item_Type_Grocery', 'Item_Type_Home Goods', 
              'Urban_Rural_Rural', 'Urban_Rural_Urban']

# load input tensor
x_1 = np.load('/workspace/DSP/data/INPUT/clsf/zero_clsf_w15_input_x_1.npy')
x_2 = np.load('/workspace/DSP/data/INPUT/clsf/zero_clsf_w15_input_x_2.npy')
x_zero_1 = np.load('/workspace/DSP/data/INPUT/clsf/zero_clsf_w15_input_x_zero_456_1.npy')
x_zero_2 = np.load('/workspace/DSP/data/INPUT/clsf/zero_clsf_w15_input_x_zero_456_2.npy')

y_nonzero = np.load('/workspace/DSP/data/INPUT/clsf/zero_clsf_w15_input_y.npy')
y_zero = np.load('/workspace/DSP/data/INPUT/clsf/zero_clsf_w15_input_y_zero_456.npy')

x_nonzero = np.concatenate([x_1, x_2])
x_zero = np.concatenate([x_zero_1, x_zero_2])

train_x = np.concatenate([x_nonzero[:int(x_nonzero.shape[0]*split_frac)], x_zero[:int(x_zero.shape[0]*split_frac)]])
train_y = np.concatenate([y_nonzero[:int(y_nonzero.shape[0]*split_frac)], y_zero[:int(y_zero.shape[0]*split_frac)]])
eval_x = np.concatenate([x_nonzero[int(x_nonzero.shape[0]*split_frac):], x_zero[int(x_zero.shape[0]*split_frac):]])
eval_y = np.concatenate([y_nonzero[int(y_nonzero.shape[0]*split_frac):], y_zero[int(y_zero.shape[0]*split_frac):]])
test_x = data_df[FEAT_KEY].to_numpy().reshape(ITEM_STORES, MONTHS, len(FEAT_KEY))[:, -window_size:, :]

# numpy to tensor
trainX_tensor = torch.FloatTensor(train_x).to(device)
trainY_tensor = torch.FloatTensor(train_y).to(device)

evalX_tensor = torch.FloatTensor(eval_x).to(device)
evalY_tensor = torch.FloatTensor(eval_y).to(device)

testX_tensor = torch.FloatTensor(test_x).to(device)

# dataset, dataloader
train_dataset = TensorDataset(trainX_tensor, trainY_tensor)
eval_dataset = TensorDataset(evalX_tensor, evalY_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# model
model = GPT_MLP_MODEL(in_features=in_features, 
                    out_features=out_features, 
                    window_size=window_size, 
                    inter_dim=inter_dim, 
                    n_heads=n_heads, 
                    n_layers=n_layers).to(device)

# criterion = nn.MSELoss().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train_hist = np.zeros(epochs)
train_losses = []
eval_losses = []
count = 0

# train, eval
for epoch in tqdm(range(epochs)):

    correct = 0
    model.train()
    for batch_idx, samples in enumerate(train_loader):
        x_train, y_train = samples
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        pred = (torch.sigmoid(outputs) > 0.5).float()
        correct += (pred == y_train).sum()

    train_acc = correct / len(train_dataset)
    print(f'train_acc : {train_acc}')

    correct = 0
    model.eval()
    for x_eval, y_eval in eval_loader:
        outputs = model(x_eval)
        loss = criterion(outputs, y_eval)
        eval_losses.append(loss.item())
        pred = (torch.sigmoid(outputs) > 0.5).float()
        correct += (pred == y_eval).sum()

    val_acc = correct /len(eval_dataset)
    print(f'val_acc : {val_acc}')
        
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

    eval_df.to_csv(f'/workspace/DSP/result/unbiased/gpt/zero_clsf_1e3_w15_4_4_56_{epochs}_eval.csv', index=None)
    test_df.to_csv(f'/workspace/DSP/result/unbiased/gpt/zero_clsf_1e3_w15_4_4_56_{epochs}_test.csv', index=None)
    
