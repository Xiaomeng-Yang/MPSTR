import torch
from typing import Optional
import math
from torch import nn as nn, Tensor
from torch.nn.modules import transformer
from torch.nn import functional as F
from timm.models.vision_transformer import VisionTransformer, PatchEmbed
from .soft_attention import SoftMaskAttention


class DecoderEdit(nn.Module):
    """
    Parallel Decoder
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='gelu',
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

    def forward(self, predicts_embed, feature_map, key_padding_mask, attn_mask):
        # mask_self_attention
        tgt2, attn = self.self_attn(predicts_embed, predicts_embed, predicts_embed,
                                    key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        tgt = self.norm1(predicts_embed + self.dropout1(tgt2))

        tgt2, attn2 = self.cross_attn(tgt, feature_map, feature_map)
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout4(tgt2))
        return tgt, attn2


class DecoderLayerSoft(nn.Module):
    """
    Transformer Decoder two-stream attention
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='gelu',
                 layer_norm_eps=1e-5):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.self_attn = SoftMaskAttention(d_model, nhead, dropout=dropout)
        '''
        change the self-attention to soft masked attention
        '''
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_q = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_c = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = transformer._get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.gelu
        super().__setstate__(state)

    def forward_stream(self, tgt: Tensor, tgt_norm: Tensor, tgt_kv: Tensor, memory: Tensor, tgt_mask: Optional[Tensor],
                       tgt_key_padding_mask: Optional[Tensor]):
        """Forward pass for a single stream (i.e. content or query)
        tgt_norm is just a LayerNorm'd tgt. Added as a separate parameter for efficiency.
        Both tgt_kv and memory are expected to be LayerNorm'd too.
        memory is LayerNorm'd by ViT.
        """
        tgt2, sa_weights = self.self_attn(tgt_norm, tgt_kv, tgt_kv, attn_mask=tgt_mask,
                                          key_padding_mask=tgt_key_padding_mask, return_attn_weights=True)
        # print(sa_weights[0][0])
        tgt = tgt + self.dropout1(tgt2)

        tgt2, ca_weights = self.cross_attn(self.norm1(tgt), memory, memory)
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(tgt)))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, sa_weights, ca_weights

    def forward(self, query, content, memory, query_mask: Optional[Tensor] = None, content_mask: Optional[Tensor] = None,
                content_key_padding_mask: Optional[Tensor] = None, update_content: bool = True):
        query_norm = self.norm_q(query)
        content_norm = self.norm_c(content)
        query = self.forward_stream(query, query_norm, content_norm, memory, query_mask, content_key_padding_mask)[0]
        if update_content:
            content = self.forward_stream(content, content_norm, content_norm, memory, content_mask,
                                          content_key_padding_mask)[0]
        return query, content


class PadPrediction(nn.Module):
    def __init__(self, embed_dim, nhead, dim_feedforward=2048, dropout=0.1,
                 activation='gelu', layer_norm_eps=1e-5):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm_q = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = transformer._get_activation_fn(activation)

        # classification head
        self.pad_cls = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.Linear(128, 2)
        )

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.gelu
        super().__setstate__(state)

    def forward(self, query, memory):
        query_norm = self.norm_q(query)
        tgt2, _ = self.cross_attn(query_norm, memory, memory)
        tgt = self.norm1(query + self.dropout1(tgt2))

        tgt2, _ = self.self_attn(tgt, tgt, tgt)
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(tgt)))))
        output = tgt + self.dropout3(tgt2)
        output = self.pad_cls(output)
        return output


class DecoderLayerSelf(nn.Module):
    """
    Parallel Decoder
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='gelu',
                 layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout3 = nn.Dropout(dropout)

        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.linear4 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm4 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout2 = nn.Dropout(dropout)
        # self.dropout4 = nn.Dropout(dropout)

        self.activation = transformer._get_activation_fn(activation)

    def forward(self, query, predicts_embed, feature_map, key_padding_mask, attn_mask):
        # mask_self_attention
        outputs, sa_weights = self.self_attn(query, predicts_embed, predicts_embed,
                                             key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # print(outputs)
        outputs = self.norm1(query + self.dropout1(outputs))
        # ff
        # self_attn_outputs = self.norm2(outputs + self.dropout3(self.linear2(self.activation(self.linear1(outputs)))))
        self_attn_outputs = self.norm2(outputs + self.linear2(self.activation(self.linear1(outputs))))

        outputs2, alphas = self.cross_attn(self_attn_outputs, feature_map, feature_map)
        outputs = self.norm3(self_attn_outputs + self.dropout2(outputs2))
        # ff
        # cross_attn_outputs = self.norm4(outputs + self.dropout4(self.linear4(self.activation(self.linear3(outputs)))))
        cross_attn_outputs = self.norm4(outputs + self.linear4(self.activation(self.linear3(outputs))))
        return cross_attn_outputs, alphas
        
        
class DecoderSelf(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm):
        super().__init__()
        self.layers = transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt_query, tgt_token, feature_map, key_padding_mask, attn_mask):
        for i, mod in enumerate(self.layers):
            predicts, alphas = mod(tgt_query, tgt_token, feature_map, key_padding_mask, attn_mask)
        return predicts


class DecoderLayer(nn.Module):
    """
    Transformer Decoder two-stream attention
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='gelu',
                 layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_q = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_c = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = transformer._get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.gelu
        super().__setstate__(state)

    def forward_stream(self, tgt: Tensor, tgt_norm: Tensor, tgt_kv: Tensor, memory: Tensor, tgt_mask: Optional[Tensor],
                       tgt_key_padding_mask: Optional[Tensor]):
        """Forward pass for a single stream (i.e. content or query)
        tgt_norm is just a LayerNorm'd tgt. Added as a separate parameter for efficiency.
        Both tgt_kv and memory are expected to be LayerNorm'd too.
        memory is LayerNorm'd by ViT.
        """
        tgt2, sa_weights = self.self_attn(tgt_norm, tgt_kv, tgt_kv, attn_mask=tgt_mask,
                                          key_padding_mask=tgt_key_padding_mask)
        # print(sa_weights[0][0])
        tgt = tgt + self.dropout1(tgt2)

        tgt2, ca_weights = self.cross_attn(self.norm1(tgt), memory, memory)
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(tgt)))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, sa_weights, ca_weights

    def forward(self, query, content, memory, query_mask: Optional[Tensor] = None, content_mask: Optional[Tensor] = None,
                content_key_padding_mask: Optional[Tensor] = None, update_content: bool = True):
        query_norm = self.norm_q(query)
        content_norm = self.norm_c(content)
        query = self.forward_stream(query, query_norm, content_norm, memory, query_mask, content_key_padding_mask)[0]
        if update_content:
            content = self.forward_stream(content, content_norm, content_norm, memory, content_mask,
                                          content_key_padding_mask)[0]
        return query, content


class Decoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm):
        super().__init__()
        self.layers = transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, query, content, memory, query_mask: Optional[Tensor] = None, content_mask: Optional[Tensor] = None,
                content_key_padding_mask: Optional[Tensor] = None):
        for i, mod in enumerate(self.layers):
            last = i == len(self.layers) - 1
            query, content = mod(query, content, memory, query_mask, content_mask, content_key_padding_mask,
                                 update_content=not last)
        query = self.norm(query)
        return query


class TokenEmbedding(nn.Module):

    def __init__(self, charset_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(charset_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens: torch.Tensor):
        return math.sqrt(self.embed_dim) * self.embedding(tokens)


class Encoder(VisionTransformer):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 class_token=False, qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed):
        super().__init__(img_size, patch_size, in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, embed_layer=embed_layer,
                         num_classes=0, global_pool='', class_token=class_token)  # these disable the classifier head

        def forward(self, x):
            # Return all tokens
            return self.forward_features(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
        
        
class DAFF(nn.Module):
    def __init__(self, in_features, img_size, patch_size, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.1, kernel_size=3, with_bn=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.h = img_size[0] // patch_size[0]
        self.w = img_size[1] // patch_size[1]
        # pointwise
        self.conv1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1, padding=0)
        # depthwise
        self.conv2 = nn.Conv2d(
            hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
            padding=(kernel_size - 1) // 2, groups=hidden_features)
        
        # pointwise
        self.conv3 = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1, padding=0)
        self.act = act_layer()
        
        self.bn1 = nn.BatchNorm2d(hidden_features)
        self.bn2 = nn.BatchNorm2d(hidden_features)
        self.bn3 = nn.BatchNorm2d(out_features)
        
        # The reduction ratio is always set to 4
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Linear(in_features, in_features//4)
        self.excitation = nn.Linear(in_features//4, in_features)
                
    def forward(self, x):
        B, N, C = x.size()
        cls_token, tokens = torch.split(x, [1, N - 1], dim=1)   # 384 * 128 * 384
        x = tokens.reshape(B, self.h, self.w, C).permute(0, 3, 1, 2)    # B, C, h, w

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        shortcut = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = shortcut + x

        x = self.conv3(x)
        x = self.bn3(x)

        weight = self.squeeze(x).flatten(1).reshape(B, 1, C)
        weight = self.excitation(self.act(self.compress(weight)))
        cls_token = cls_token * weight
        
        tokens = x.flatten(2).permute(0, 2, 1)
        out = torch.cat((cls_token, tokens), dim=1)
        return out


'''
class Improve_Encoder(Edit_ViT):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed,
                 distilled=False, representation_size=None, norm_layer=None, act_layer=None, weight_init='',
                 apply_dhvt=True):
        super().__init__(img_size, patch_size, in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, embed_layer=embed_layer, distilled=distilled,
                         representation_size=representation_size, norm_layer=norm_layer, act_layer=act_layer,
                         weight_init=weight_init, apply_dhvt=apply_dhvt)
        
        def forward(self, x):
            return self.forward_features(x)
'''

