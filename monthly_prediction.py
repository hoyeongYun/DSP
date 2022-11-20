import pandas as pd
import numpy as np
np.set_printoptions(linewidth=np.inf)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import transformers
from model.monthly_prediction_model import Monthly_Predicter
from dataset.monthly_prediction_dataset import Monthly_Prediction_Dataset
from dataset.data_preprocessing import load_data, create_monthly_prediction_df, split_monthly_prediction_df
from tqdm import tqdm

# args
torch.manual_seed(21)
DATA_PATH = '/workspace/DSP/data/PREPROCESSED/Data_Total.csv'
device = torch.device('cuda')
split_strategy = 2
batch_size = 4
lr = 1e-4
epochs = 5
in_features = 7
inter_dim = 512
output_len=1

# load data
data_monthly_df, test_df, preprocs = create_monthly_prediction_df(load_data(DATA_PATH))
# split data
train_df, eval_df = split_monthly_prediction_df(df=data_monthly_df, strategy=split_strategy, preprocs=preprocs)
# dataset
train_dataset = Monthly_Prediction_Dataset(df=train_df, device=device)
eval_dataset = Monthly_Prediction_Dataset(df=eval_df, device=device)
test_dataset = Monthly_Prediction_Dataset(df=test_df, device=device, mode='test')
# dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# print(train_dataset[4][0].shape)
# print(train_dataset[4][1].shape)
model = Monthly_Predicter(in_features=in_features, inter_dim=inter_dim, output_len=output_len).to(device)
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_losses = []
eval_losses = []
avg_train_losses = []
avg_eval_losses = []
 
for epoch in tqdm(range(epochs)):
    avg_cost = 0.0

    model.train()
    for batch_idx, samples in enumerate(train_loader):
        x_train, y_train = samples
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    for x_eval, y_eval in eval_loader:
        outputs = model(x_eval)
        loss = criterion(outputs, y_eval)
        eval_losses.append(loss.item())

    train_loss = np.average(train_losses)
    eval_loss = np.average(eval_losses)
    avg_train_losses.append(train_loss)
    avg_eval_losses.append(eval_loss)
    
    epoch_len = len(str(epoch))
    print(f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {eval_loss:.5f}')

# 642, 88, 1

with torch.no_grad():
    result = model(test_dataset.x, output_len=3)
    result = result.cpu().detach().numpy()

result = result.reshape(642, -1)
result_df = pd.DataFrame(result)
result_df.to_csv('/workspace/DSP/result/2022_1120_1.csv', index=None)



