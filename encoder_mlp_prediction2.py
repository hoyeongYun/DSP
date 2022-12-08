import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import gc
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.utils.data import DataLoader, TensorDataset
from model.end_to_end_model import End_To_End_Predicter

np.set_printoptions(linewidth=np.inf)
torch.manual_seed(42)

data_df = pd.read_csv('/workspace/DSP/data/PREPROCESSED/End_To_End_Data.csv')
data_total_df = pd.read_csv('/workspace/DSP/data/PREPROCESSED/Data_Total.csv')
item_priority_df = pd.read_csv('/workspace/DSP/data/PREPROCESSED/Item_Priority.csv')
item_store_df = pd.read_csv('/workspace/DSP/data/PREPROCESSED/Item_Store_Data.csv')
aug_0_df = pd.read_csv('/workspace/DSP/data/PREPROCESSED/Augmented_Data_0.csv')
aug_1_df = pd.read_csv('/workspace/DSP/data/PREPROCESSED/Augmented_Data_1.csv')
aug_2_df = pd.read_csv('/workspace/DSP/data/PREPROCESSED/Augmented_Data_2.csv')
aug_3_df = pd.read_csv('/workspace/DSP/data/PREPROCESSED/Augmented_Data_3.csv')
augmented_df = pd.concat([aug_0_df, aug_1_df, aug_2_df, aug_3_df]).reset_index(drop=True)

req_cols = ['Month','Augmented_Industry_Size', 'Sales_At_The_Month', 'Is_Zero', 'Augmented_Retail_Size']
train_eval_df = augmented_df[req_cols].copy()
month_scaler = MinMaxScaler()
# industry_scaler = MinMaxScaler()
# retail_scaler = MinMaxScaler()
train_eval_df['Month'] = month_scaler.fit_transform(train_eval_df['Month'].to_numpy().reshape(-1, 1))
# train_eval_df['Augmented_Industry_Size'] = month_scaler.fit_transform(train_eval_df['Augmented_Industry_Size'].to_numpy().reshape(-1, 1))
# train_eval_df['Augmented_Retail_Size'] = month_scaler.fit_transform(train_eval_df['Augmented_Retail_Size'].to_numpy().reshape(-1, 1))

class Encoder_MLP_Dataset(torch.utils.data.Dataset):
    def __init__(self, df, x_cols=5, total_month=861, total_item=11556, input_window=840, start=0, test=False):
        
        orig_data = df.to_numpy().reshape(total_item, total_month, x_cols)
        self.test = test
        self.x = torch.FloatTensor(orig_data[:, start:start+input_window, :])
        if not self.test:
            self.y = torch.FloatTensor(orig_data[:, -input_window:, :])
        self.len = len(self.x)        

    def __getitem__(self, i):
        if self.test:
            return self.x[i]
        else:
            return self.x[i], self.y[i]
    def __len__(self):
        return self.len


def weighted_mse_loss(result, target, device):
    w_zero = 1
    max_target_val = 135.2
    priority_to_non_zero_values = 500
    weight = torch.FloatTensor(np.where(target.cpu().detach().numpy() < 1.0, w_zero, priority_to_non_zero_values + np.exp(target.cpu().detach().numpy() / max_target_val))).to(device)
    return torch.sum(weight * (result - target) ** 2)

train_dataset = Encoder_MLP_Dataset(train_eval_df)
test_dataset = Encoder_MLP_Dataset(train_eval_df, start=21, test=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=12)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=12)
device = torch.device("cuda")

model = End_To_End_Predicter(in_features=5, out_features=1, inter_dim=512).to(device)
lr = 1e-2
criterion = weighted_mse_loss
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_losses = []
avg_train_losses = []
epochs = 500
for epoch in tqdm(range(epochs)):

    model.train()
    gc.collect()
    torch.cuda.empty_cache()
    for x_train, y_train in train_loader:
        # print(x_train.shape)      # 6 840 5
        # print(y_train.shape)      # 6 840 1
        result = model(x_train.to(device))
        # print(result.shape)       # 6 840 1
        loss = criterion(result, y_train.to(device), device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    train_loss = np.average(train_losses)

    epoch_len = len(str(epoch))
    print()
    print(f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +  f'train_loss: {train_loss:.8f}')          

    if (epoch+1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            result_df = pd.DataFrame()
            for x_test in test_loader:
                output = model(x_test.to(device), test=True)
                output_df = pd.DataFrame(output)
                result_df = pd.concat([result_df, output_df])
            result_df.to_csv(f'/workspace/DSP/result/em/encoder_mlp_epoch{epoch+1}.csv', index=None)
        