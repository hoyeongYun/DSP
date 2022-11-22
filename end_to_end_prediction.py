import pandas as pd
import numpy as np
np.set_printoptions(linewidth=np.inf)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import transformers
# from model.monthly_prediction_model import Monthly_Predicter
from utils.utils import load_data
from model.end_to_end_model import End_To_End_Predicter
from dataset.end_to_end_dataset import End_To_End_Dataset
# from dataset.monthly_prediction_dataset import Monthly_Prediction_Dataset
from dataset.end_to_end_preprocessing import create_ete_prediction_df, train_eval_df_split
from tqdm import tqdm

# args
torch.manual_seed(21)
DATA_PATH = '/workspace/DSP/data/PREPROCESSED/End_To_End_Data.csv'
device = torch.device('cuda')
split_strategy = 2
batch_size = 4
lr = 1e-3
epochs = 100
in_features = 8*18
out_features = 18
inter_dim = 1024
output_len=1

# load data
data_df = load_data(DATA_PATH)
train_eval_df, test_df, preprocs = create_ete_prediction_df(data_df)
# split data
train_df, eval_df = train_eval_df_split(df=train_eval_df, strategy=split_strategy, preprocs=preprocs)
# dataset
train_dataset = End_To_End_Dataset(df=train_eval_df, device=device)
# eval_dataset = End_To_End_Dataset(df=eval_df, device=device)
test_dataset = End_To_End_Dataset(df=test_df, device=device, mode='test')
# dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
# eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = End_To_End_Predicter(in_features=in_features, out_features=out_features, inter_dim=inter_dim, output_len=output_len).to(device)
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_losses = []
# eval_losses = []
avg_train_losses = []
# avg_eval_losses = []

for epoch in tqdm(range(epochs)):
    model.train()
    for x_train, y_train, store_ids in train_loader:
        outputs = model(x_train, store_ids)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # model.eval()
    # for x_eval, y_eval, store_ids in eval_loader:
    #     outputs = model(x_eval, store_ids)
    #     loss = criterion(outputs, y_eval)
    #     eval_losses.append(loss.item())

    train_loss = np.average(train_losses)
    # eval_loss = np.average(eval_losses)
    
    epoch_len = len(str(epoch))
    print()
    print(f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.8f}')
                    #   ' + f'valid_loss: {eval_loss:.5f}')

with torch.no_grad():
    result = model(test_dataset.x, test_dataset.store_ids, test=True)        # 642, 88, 1
    result = result.cpu().detach().numpy()

result = result.reshape(642, -1)
result_df = pd.DataFrame(result)
result_df.to_csv('/workspace/DSP/result/ete/full_train_1e3_100_ver.csv', index=None)



