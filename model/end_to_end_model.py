import torch
import torch.nn as nn
import transformers
import math

class StoreID_Encoding(nn.Module):
    def __init__(self, input_dim, dropout=0.0, store_max_len=642, total_months=84):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.input_dim = input_dim
        self.store_max_len = store_max_len
        self.total_months = total_months
        se = torch.zeros(store_max_len, input_dim)
        position = torch.arange(0, store_max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.input_dim, 2).float() * (-math.log(10000.0) / self.input_dim))
        se[:, 0::2] = torch.sin(position * div_term)
        se[:, 1::2] = torch.cos(position * div_term)
        se = se.reshape(self.store_max_len, 1, input_dim) * torch.ones(self.store_max_len, total_months, input_dim)
        self.register_buffer('se', se)

    def forward(self, x, store_ids):
        store_ids -= torch.ones_like(store_ids, dtype=torch.long, device=x.device)      # to index
        x = x + self.se[store_ids.cpu().numpy(), :, :]
        return self.dropout(x)

class End_To_End_Predicter(nn.Module):
    """gpt2 model for monthly prediction task"""
    def __init__(
        self,
        in_features: int,       # 18D
        out_features: int,      # 18
        inter_dim: int,   # 1024
        output_len: int = 1):
        super().__init__()
        self.encoder = nn.Linear(in_features, inter_dim, bias=False)     # ex) B, 84, 18D -> 1024
        self.decoder = nn.Linear(inter_dim, 1, bias=False)     # ex) B, 84, 1024 -> 18
        self.gpt_model = transformers.GPT2Model(
            transformers.GPT2Config(n_embd=inter_dim,
                                    vocab_size=in_features,
                                    use_cache=True,
                                    n_head=8,
                                    n_layer=8))
        del self.gpt_model.wte
        self.output_len = output_len
        self.inter_dim = inter_dim
        self.in_features = in_features
        self.store_id_encoder = StoreID_Encoding(inter_dim)

    def forward(self, feats, store_ids, output_len=1, test=False):
        """
        Args:
            feats: input tensor                 EX) [B, 84, 18D]
            target_shape: output shape          EX) [B, 84, 18]
            store_ids: [1, 2, 4, 5]
            refactoring 필요
        """
        inp_feats = feats
        feats = self.encoder(feats)         # [B, 84, 18D] -> [B, 84, 1024]
        feats = self.store_id_encoder(feats, store_ids)
        past = None
        all_outputs = []
        all_outputs_decoded = []
        for output_id in range(output_len):
            pred_so_far = sum([el.size(1) for el in all_outputs])
            position_ids = torch.arange(pred_so_far,
                                        pred_so_far + feats.size(1),
                                        dtype=torch.long,
                                        device=feats.device)
            if test:
                position_ids = position_ids + (torch.ones_like(position_ids, dtype=torch.long, device=feats.device)*3)
            outputs = self.gpt_model(inputs_embeds=feats,               
                                     past_key_values=past,           
                                     position_ids=position_ids)
            last_hidden_state = outputs.last_hidden_state
            past = outputs.past_key_values
            all_outputs.append(last_hidden_state)
            all_outputs_decoded.append(self.decoder(last_hidden_state))
            feats = last_hidden_state[:, -1:, :]
        all_outputs = torch.cat(all_outputs, dim=1)
        all_outputs_decoded = torch.cat(all_outputs_decoded, dim=1)
        prev = inp_feats
        all_outputs = all_outputs_decoded
        return all_outputs

            



