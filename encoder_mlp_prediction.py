import pandas as pd
import numpy as np
from tqdm import tqdm
import math

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.utils.data import DataLoader, TensorDataset
np.set_printoptions(linewidth=np.inf)

data_df = pd.read_csv('/workspace/DSP/data/PREPROCESSED/End_To_End_Data.csv')
data_total_df = pd.read_csv('/workspace/DSP/data/PREPROCESSED/Data_Total.csv')
item_priority_df = pd.read_csv('/workspace/DSP/data/PREPROCESSED/Item_Priority.csv')
item_store_df = pd.read_csv('/workspace/DSP/data/PREPROCESSED/Item_Store_Data.csv')
aug_0_df = pd.read_csv('/workspace/DSP/data/PREPROCESSED/Augmented_Data_0.csv')
aug_1_df = pd.read_csv('/workspace/DSP/data/PREPROCESSED/Augmented_Data_1.csv')
aug_2_df = pd.read_csv('/workspace/DSP/data/PREPROCESSED/Augmented_Data_2.csv')
aug_3_df = pd.read_csv('/workspace/DSP/data/PREPROCESSED/Augmented_Data_3.csv')
augmented_df = pd.concat([aug_0_df, aug_1_df, aug_2_df, aug_3_df]).reset_index(drop=True)

req_cols = ['Month', 'Augmented_Industry_Size', 'Sales_At_The_Month', 'Is_Zero', 'Augmented_Retail_Size']
train_eval_df = augmented_df[req_cols].copy()
month_scaler = MinMaxScaler()
industry_scaler = MinMaxScaler()
retail_scaler = MinMaxScaler()
train_eval_df['Month'] = month_scaler.fit_transform(train_eval_df['Month'].to_numpy().reshape(-1, 1))
# train_eval_df['Augmented_Industry_Size'] = month_scaler.fit_transform(train_eval_df['Augmented_Industry_Size'].to_numpy().reshape(-1, 1))
# train_eval_df['Augmented_Retail_Size'] = month_scaler.fit_transform(train_eval_df['Augmented_Retail_Size'].to_numpy().reshape(-1, 1))

class Window_Dataset(torch.utils.data.Dataset):
    def __init__(self, df, x_cols=5, y_cols=1, total_month=861, total_item=11556, input_window=120, output_window=21, stride=30):
        
        orig_data = df.to_numpy().reshape(total_item, total_month, x_cols)
        self.num_samples_per_item = (total_month - input_window - output_window) // stride + 1

        X = np.zeros([total_item * self.num_samples_per_item, input_window, x_cols])
        Y = np.zeros([total_item * self.num_samples_per_item, output_window, y_cols])
        
        for item in range(total_item):
            for i in range(self.num_samples_per_item):
                start_x = stride*i
                end_x = start_x + input_window
                X[(item*self.num_samples_per_item + i), :, :] = orig_data[item, start_x:end_x, :]

                start_y = stride*i + input_window
                end_y = start_y + output_window
                Y[(item*self.num_samples_per_item + i), :, :] = orig_data[item, start_y:end_y, -1:]


        self.test_x = torch.tensor(orig_data[:, -input_window:, :])
        self.x = X
        self.y = Y
        self.len = total_item * self.num_samples_per_item

        assert self.x.shape == (total_item * self.num_samples_per_item, input_window, x_cols)
        assert self.y.shape == (total_item * self.num_samples_per_item, output_window, y_cols)

    def __getitem__(self, i):
        return self.x[i], self.y[i]
    def __len__(self):
        return self.len


class Encoder_MLP_Model(nn.Module):
    def __init__(self, iw, ow, d_model, nhead, nlayers, dropout=0.5):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers) 
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.encoder = nn.Sequential(
            nn.Linear(5, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, d_model)
        )
        
        self.linear =  nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(iw, (iw+ow)//2),
            nn.ReLU(),
            nn.Linear((iw+ow)//2, ow)
        ) 

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, srcmask):
        src = self.encoder(src) # batch, window, 5 -> batch, window , 512
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src.transpose(0,1), srcmask).transpose(0,1)
        output = self.linear(output)[:,:,0]
        output = self.linear2(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)        # 1, max_len, d_model
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[0, :x.size(1), :]
        return self.dropout(x)

# def gen_attention_mask(x):
#     mask = torch.eq(x, 0)
#     return mask

def weighted_mse_loss(result, target, device):
    w_zero = 0.05
    max_target_val = 135.2
    priority_to_non_zero_values = 1
    weight = torch.FloatTensor(np.where(target.cpu().detach().numpy() < 1.0, w_zero, priority_to_non_zero_values + np.exp(target.cpu().detach().numpy() / max_target_val))).to(device)
    return torch.sum(weight * (result - target) ** 2)

train_dataset = Window_Dataset(train_eval_df)
test_x = train_dataset.test_x
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_dataset.num_samples_per_item*4)

device = torch.device("cuda")
lr = 3e-3
model = Encoder_MLP_Model(120, 21, 512, 8, 8, 0.1).to(device)
criterion = weighted_mse_loss
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_losses = []
avg_train_losses = []
epochs = 600

for epoch in tqdm(range(epochs)):

    model.train()
    for x_train, y_train in train_loader:
        result = model(x_train.float().to(device), model.generate_square_subsequent_mask(x_train.shape[1]).to(device))
        loss = criterion(result, y_train[:,:,0].float().to(device), device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    if epoch == 4:
        with torch.no_grad():
            result = model(test_x.float().to(device), model.generate_square_subsequent_mask(test_x.shape[1]).to(device))
            result = result.cpu().detach().numpy()
            result_df = pd.DataFrame(result)
            result_df.to_csv('/workspace/DSP/result/em/encoder_mlp_epoch5.csv', index=None)

    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            result = model(test_x.float().to(device), model.generate_square_subsequent_mask(test_x.shape[1]).to(device))
            result = result.cpu().detach().numpy()
            result_df = pd.DataFrame(result)
            result_df.to_csv(f'/workspace/DSP/result/em/encoder_mlp_epoch{epoch+1}.csv', index=None)
    
    train_loss = np.average(train_losses)

    epoch_len = len(str(epoch))
    print()
    print(f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +  f'train_loss: {train_loss:.8f}')        
    # positional encoding 추가, test때 왜 모든 item 값이 같은지 trouble shooting