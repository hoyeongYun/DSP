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
from dataset.dataset.ete_per_item_dataset import ETE_Per_Item_Dataset
# from dataset.monthly_prediction_dataset import Monthly_Prediction_Dataset
from dataset.preprocessing.ete_per_item_preprocessing import get_model_keys, create_per_item_prediction_df, get_df_by_model_type
from tqdm import tqdm

# args
torch.manual_seed(21)
DATA_PATH = '/workspace/DSP/data/PREPROCESSED/End_To_End_Data.csv'
device = torch.device('cuda')
# split_strategy = 2
batch_size = 4
lr = 1e-3
epochs = 400
# in_features = 26
out_features = 1
inter_dim = 512
output_len=1

# data preprocessing
data_df = load_data(DATA_PATH)
model_keys = get_model_keys(data_df)
df_type_0, df_type_1 = get_df_by_model_type(data_df, model_keys)
train_eval_0_df, test_0_df, preprocs_0 = create_per_item_prediction_df(df_type_0)
train_eval_1_df, test_1_df, preprocs_1 = create_per_item_prediction_df(df_type_1)

# dataset
train_0_dataset = ETE_Per_Item_Dataset(df=train_eval_0_df, device=device)
train_1_dataset = ETE_Per_Item_Dataset(df=train_eval_1_df, device=device)
test_0_dataset = ETE_Per_Item_Dataset(df=test_0_df, device=device, mode='test')
test_1_dataset = ETE_Per_Item_Dataset(df=test_1_df, device=device, mode='test')

# dataloader
train_0_loader = DataLoader(train_0_dataset, batch_size=batch_size, shuffle=False)
train_1_loader = DataLoader(train_1_dataset, batch_size=batch_size, shuffle=False)

# model
model_0 = End_To_End_Predicter(in_features=train_0_dataset.req_cols, out_features=out_features, inter_dim=inter_dim, output_len=output_len).to(device)
model_1 = End_To_End_Predicter(in_features=train_1_dataset.req_cols, out_features=out_features, inter_dim=inter_dim, output_len=output_len).to(device)
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model_0.parameters(), lr=lr)

train_losses = []
avg_train_losses = []

for epoch in tqdm(range(epochs)):
    model_0.train()
    for x_train, y_train, store_ids in train_0_loader:
        outputs = model_0(x_train, store_ids)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    train_loss = np.average(train_losses)
    
    print()
    print(f'[{epoch:>{len(str(epoch))}}/{epochs:>{len(str(epoch))}}] ' +
                     f'train_loss: {train_loss:.8f}')

with torch.no_grad():
    result = model_0(test_0_dataset.x, test_0_dataset.store_ids, test=True)        # 351, 84, 1
    result = result.cpu().detach().numpy()

result = result.reshape(351, -1)
result_df = pd.DataFrame(result)
result_df.to_csv('/workspace/DSP/result/per_item/1e3_400_ver.csv', index=None)



