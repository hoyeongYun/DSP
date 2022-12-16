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

# args
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MONTHS = 87
STORES = 642
ITEM_STORES = 11556
in_features = 9
out_features = 3
inter_dim = 56          ###
window_size = 18
n_heads = 4             ##
n_layers = 4            #
split_frac = 0.95       #
lr = 1e-3               #
epochs = 100            #
batch_size = 48
count = 0
patience = 9


# load data
x = np.load('/workspace/DSP/data/INPUT/final/final_input_187629_18_9_x.npy')
y = np.load('/workspace/DSP/data/INPUT/final/final_input_187629_18_3_y.npy')
# y = y[:, -1, :]
# test input # test 새로 생성 해야 됨
test_x = np.load('/workspace/DSP/data/INPUT/final/test_input_x_final.npy')  # 11556, 18, 9

# train eval split
train_x = x[:int(len(x)*split_frac)]
train_y = y[:int(len(y)*split_frac)]
eval_x = x[int(len(x)*split_frac):]
eval_y = y[int(len(y)*split_frac):]

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
criterion = weighted_mse_loss
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train_hist = np.zeros(epochs)
train_losses = []
eval_losses = []

# train, eval
for epoch in tqdm(range(epochs)):

    model.train()
    for x_train, y_train in train_loader:
        outputs = model(x_train)
        loss = criterion(outputs, y_train, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
    model.eval()
    with torch.no_grad():
        for x_eval, y_eval in eval_loader:
            outputs = model(x_eval)
            loss = criterion(outputs, y_eval, device)
            eval_losses.append(loss.item())
    
    train_loss = np.average(train_losses)
    eval_loss = np.average(eval_losses)
    epoch_len = len(str(epoch))

    print('\n')
    print(f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' + f'train_loss: {train_loss:.5f} ' + f'valid_loss: {eval_loss:.5f}')

    if train_hist[epoch-1] < train_hist[epoch]:
        count += 1
        print(count)
    if count >= patience:
        print('\nEarly Stopping')
        break
# output result
model.eval()
with torch.no_grad():
    eval_result = model(evalX_tensor).detach().cpu().numpy()        # B x 18 x 3        or B x 3
    test_result = model(testX_tensor).detach().cpu().numpy()        # 11556 x 18 x 3    or 11556 x 3

    # for ver_1
    eval_df = pd.DataFrame(eval_result[:, -1, :].reshape(-1, out_features))       # 코랩에 eval_y 3개월 합친 거 vs 예측 3개월치 합친거 비교 코드 만들기 이때 각 월에서 int로 바꾸고 합칠건지 합치고 바꿀건지 결정도 해야됨
    test_df = pd.DataFrame(test_result[:, -1, :].reshape(-1, out_features))

    # for ver_2
    # eval_df = pd.DataFrame(eval_result)       # 코랩에 eval_y 3개월 합친 거 vs 예측 3개월치 합친거 비교 코드 만들기 이때 각 월에서 int로 바꾸고 합칠건지 합치고 바꿀건지 결정도 해야됨
    # test_df = pd.DataFrame(test_result)

    eval_df.to_csv(f'/workspace/DSP/result/final/ver_1/{lr}_{split_frac}_{inter_dim}_{n_heads}_{eval_loss:.5f}_{epochs}_eval.csv', index=None)
    test_df.to_csv(f'/workspace/DSP/result/final/ver_1/{lr}_{split_frac}_{inter_dim}_{n_heads}_{epochs}_test.csv', index=None)
    
