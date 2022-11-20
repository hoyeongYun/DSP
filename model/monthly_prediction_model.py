import torch
import torch.nn as nn
import transformers

class Monthly_Predicter(nn.Module):
    """gpt2 model for monthly prediction task"""
    def __init__(
        self,
        in_features: int,       # 7
        inter_dim: int,   # 512
        output_len: int = 1):
        super().__init__()
        self.encoder = nn.Linear(in_features, inter_dim, bias=False)     # ex) B, 86, 7 -> 512
        self.decoder = nn.Linear(inter_dim, 1, bias=False)     # ex) B, 86, 512 -> 1
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


    def forward(self, feats, output_len=1):
        """
        Args:
            feats: input tensor (B, total_month, # of feature)      EX) [B, 86, 7]
            target_shape: output shape (B, total_month, 1)          EX) [B, 86, 1]
        """
        full_inp_feats = feats
        # full_orig_feats = feats
        inp_feats = full_inp_feats
        # orig_feats_len = feats.size(1)
        feats = self.encoder(feats)         # [B, 86, 7] -> [B, 86, 512]
        # orig_feats_encoded = feats
        past = None
        all_outputs = []
        all_outputs_decoded = []
        for output_id in range(output_len):
            pred_so_far = sum([el.size(1) for el in all_outputs])
            position_ids = torch.arange(pred_so_far,
                                        pred_so_far + feats.size(1),
                                        dtype=torch.long,
                                        device=feats.device)
            
            # print('gg')
            # print(position_ids)
            outputs = self.gpt_model(inputs_embeds=feats,               # [8, 86, 512]
                                     past_key_values=past,              # 다음 sequence 계산할때 속도 높히기 위해 이전 output으로 나온 keyvalue를 다시 넣어줌
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

            



