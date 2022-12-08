import torch
import torch.nn as nn
import transformers
import math

class GPT_MLP_MODEL(nn.Module):
    """gpt2 model for monthly prediction task"""
    def __init__(
        self,
        in_features: int,       
        out_features: int,
        window_size: int,      
        inter_dim: int,
        n_heads: int,
        n_layers: int,   
        output_len: int = 1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_features, (in_features + inter_dim)//2),
            nn.ReLU(),
            nn.Linear((in_features + inter_dim)//2, inter_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(inter_dim, (inter_dim + out_features)//2),
            nn.ReLU(),
            nn.Linear((inter_dim + out_features)//2, out_features)
        )
        self.gpt_model = transformers.GPT2Model(
            transformers.GPT2Config(n_embd=inter_dim,
                                    vocab_size=in_features,
                                    use_cache=True,
                                    n_head=n_heads,
                                    n_layer=n_layers))
        del self.gpt_model.wte
        self.output_len = output_len
        self.inter_dim = inter_dim
        self.in_features = in_features
        self.window_size = window_size

    def forward(self, feats, output_len=1):
        feats = self.encoder(feats)         # [B, window, in_fetures] -> [B, window, inter_dim]
        position_ids = torch.arange(self.window_size, dtype=torch.long, device=feats.device)
        outputs = self.gpt_model(inputs_embeds=feats,               
                                past_key_values=None,           
                                position_ids=position_ids)
        output = self.decoder(outputs.last_hidden_state[:, -1, :])        # [B, inter_dim] -> [B, out_features]
        return output