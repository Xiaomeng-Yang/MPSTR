import torch
from torch import nn as nn
from torch.nn.modules import transformer


class AT_Decoder(nn.Module):
    """
    An Autoregressive decoder
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu',
                 layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.linear4 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm4 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = transformer._get_activation_fn(activation)

    def forward(self, char_embedding, feature_map, tgt_mask, tgt_key_padding_mask):
        # mask_self_attention
        outputs, _ = self.self_attn(char_embedding, char_embedding, char_embedding, attn_mask=tgt_mask,
                                    key_padding_mask=tgt_key_padding_mask)
        outputs = self.norm1(char_embedding + self.dropout1(outputs))
        # ff
        self_attn_outputs = self.norm2(outputs + self.linear2(self.activation(self.linear1(outputs))))

        outputs2, alphas = self.cross_attn(self_attn_outputs, feature_map, feature_map)
        outputs = self.norm3(self_attn_outputs + self.dropout2(outputs2))
        # ff
        cross_attn_outputs = self.norm4(outputs + self.linear4(self.activation(self.linear3(outputs))))
        return cross_attn_outputs
        

class AT_DecoderEdit(nn.Module):
    """
    An Autoregressive decoder
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu',
                 layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout2 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout4 = nn.Dropout(dropout)
        self.activation = transformer._get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, char_embedding, feature_map, tgt_mask, tgt_key_padding_mask):
        tgt2, attn = self.self_attn(char_embedding, char_embedding, char_embedding,
                                    key_padding_mask=tgt_key_padding_mask, attn_mask=tgt_mask)
        tgt = self.norm1(char_embedding + self.dropout1(tgt2))

        tgt2, attn2 = self.cross_attn(tgt, feature_map, feature_map)
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout4(tgt2))
        return tgt

