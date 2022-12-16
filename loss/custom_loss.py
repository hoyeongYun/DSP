import torch
import numpy as np

def weighted_mse_loss(result, target, device):
    # w_zero = 2
    max_target_val = 145.0
    # weight = torch.FloatTensor(np.where(target.cpu().detach().numpy() < 1.0, w_zero, np.exp(target.cpu().detach().numpy() / max_target_val))).to(device)
    weight = torch.FloatTensor(1+np.exp(target.cpu().detach().numpy()/max_target_val)).to(device)
    return torch.mean(weight * (result - target) ** 2)