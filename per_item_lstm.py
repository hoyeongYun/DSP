
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.utils.data import DataLoader, TensorDataset

# fix seed
random_seed = 21
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

# hyperparameters
window_size = 18
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr = 1e-4
epochs = 70
count = 0
patience = 100
input_dim = 9
hidden_dim = 56
output_dim = 1

# king crab
x_king_crab = np.load('/workspace/DSP/data/INPUT/per_item/0_King Crab_input_x.npy')
y_king_crab = np.load('/workspace/DSP/data/INPUT/per_item/0_King Crab_input_y.npy')

# keyboard
x_keyboard = np.load('/workspace/DSP/data/INPUT/per_item/1_Keyboard_input_x.npy')
y_keyboard = np.load('/workspace/DSP/data/INPUT/per_item/1_Keyboard_input_y.npy')

# steak
x_steak = np.load('/workspace/DSP/data/INPUT/per_item/2_Steak_input_x.npy')
y_steak = np.load('/workspace/DSP/data/INPUT/per_item/2_Steak_input_y.npy')

# mouse
x_mouse = np.load('/workspace/DSP/data/INPUT/per_item/3_Mouse_input_x.npy')
y_mouse = np.load('/workspace/DSP/data/INPUT/per_item/3_Mouse_input_y.npy')

# paint
x_paint = np.load('/workspace/DSP/data/INPUT/per_item/4_Paint_input_x.npy')
y_paint = np.load('/workspace/DSP/data/INPUT/per_item/4_Paint_input_y.npy')

# shrimp
x_shrimp = np.load('/workspace/DSP/data/INPUT/per_item/5_Shrimp_input_x.npy')
y_shrimp = np.load('/workspace/DSP/data/INPUT/per_item/5_Shrimp_input_y.npy')

# phone charger
x_phone_charger = np.load('/workspace/DSP/data/INPUT/per_item/6_Phone Charger_input_x.npy')
y_phone_charger = np.load('/workspace/DSP/data/INPUT/per_item/6_Phone Charger_input_y.npy')

# test input
test_x = np.load('/workspace/DSP/data/INPUT/test_input_w18_9_x.npy')

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
  
    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.layers, self.seq_len, self.hidden_dim),
            torch.zeros(self.layers, self.seq_len, self.hidden_dim)
        )
    def forward(self, x):
        x, _status = self.lstm(x)
        x = self.fc(x[:, -1])
        return x

def weighted_mse_loss(result, target, device):
    w_zero = 0.05
    max_target_val = 145.0
    weight = torch.FloatTensor(np.where(target.cpu().detach().numpy() < 1.0, w_zero, np.exp(target.cpu().detach().numpy() / max_target_val))).to(device)
    return torch.sum(weight * (result - target) ** 2)

# which item
item = 'King Crab' 
# item = 'Keyboard'
# item = 'Steak'
# item = 'Mouse'
# item = 'Paint'
# item = 'Shrimp'
# item = 'Phone Charger'

x = x_king_crab
y = y_king_crab

# x = x_keyboard
# y = y_keyboard

# x = x_steak
# y = y_steak

# x = x_mouse
# y = y_mouse

# x = x_paint
# y = y_paint

# x = x_shrimp
# y = y_shrimp

# x = x_phone_charger
# y = y_phone_charger


train_size = int(len(x) * 0.9)
train_x = x[:train_size]
train_y = y[:train_size]

eval_x = x[train_size:]
eval_y = y[train_size:]

# input tensor
trainX_tensor = torch.FloatTensor(train_x).to(device)
trainY_tensor = torch.FloatTensor(train_y).to(device)
evalX_tensor = torch.FloatTensor(eval_x).to(device)
evalY_tensor = torch.FloatTensor(eval_y).to(device)
testX_tensor = torch.FloatTensor(test_x).to(device)

# dataset
train_dataset = TensorDataset(trainX_tensor, trainY_tensor)
eval_dataset = TensorDataset(evalX_tensor, evalY_tensor)

# dataloader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False, drop_last=True)

# model
model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, seq_len=window_size, output_dim=output_dim, layers=1).to(device)

# criterion = nn.MSELoss().to(device)
criterion = weighted_mse_loss
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train_hist = np.zeros(epochs)

train_losses = []
eval_losses = []
last_epoch = -1
last_val_loss = -1
print(f'{item}_{lr}_{hidden_dim} case start')
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
                        f'train_loss: {train_loss:.8f} ' +
                        f'valid_loss: {eval_loss:.8f}')
    last_epoch = epoch
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

    eval_df.to_csv(f'/workspace/DSP/result/unbiased/per_item/{item}/{item}_{lr}_{hidden_dim}_{last_epoch}_{eval_loss}eval.csv', index=None)
    test_df.to_csv(f'/workspace/DSP/result/unbiased/per_item/{item}/{item}_{lr}_{hidden_dim}_{last_epoch}_test.csv', index=None)
   
