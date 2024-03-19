from functools import partial
from typing import Sequence, Any, Optional
from einops import repeat
import torch
import math
from typing import Optional, Tuple, List
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pytorch_lightning.utilities.types import STEP_OUTPUT
from timm.models.helpers import named_apply

from strhub.data.utils import PIMNetTokenizer
from strhub.models.base import BaseSystem
from strhub.models.utils import init_weights
from .modules import Encoder, TokenEmbedding
from .parallel_decoder import Decoder


class PIMNet(BaseSystem):
    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
                 img_size: Sequence[int], patch_size: Sequence[int], embed_dim: int,
                 enc_num_heads: int, enc_mlp_ratio: int, enc_depth: int, dec_num_heads: int,
                 num_iter: int, dropout: float,
                 **kwargs: Any) -> None:
        tokenizer = PIMNetTokenizer(charset_train)
        super().__init__(tokenizer, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.eos_id = tokenizer.eos_id
        self.mask_id = tokenizer.mask_id
        self.pad_id = tokenizer.pad_id

        self.save_hyperparameters()
        self.num_iter = num_iter
        self.max_len = max_label_length
        self.top_k = math.ceil(self.max_len / self.num_iter)
        self.dec_num_heads = dec_num_heads

        self.text_embed = TokenEmbedding(len(self.tokenizer), embed_dim)
        self.pos_embed = nn.Parameter(torch.Tensor(1, self.max_len, embed_dim))
        # We don't predict <pad>
        self.head = nn.Linear(embed_dim, len(self.tokenizer) - 2)   # don't predict <pad> or <mask>
        self.dropout = nn.Dropout(p=dropout)
        self.encoder = Encoder(img_size, patch_size, embed_dim=embed_dim, depth=enc_depth,
                               num_heads=enc_num_heads, mlp_ratio=enc_mlp_ratio)

        self.decoder = Decoder(d_model=embed_dim, nhead=dec_num_heads)
        # Encoder has its own init
        named_apply(partial(init_weights, exclude=['encoder']), self)
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def encode(self, img: torch.Tensor):
        return self.encoder(img)

    def decode(self, tgt: torch.Tensor, feature_map: torch.Tensor, tgt_mask: Optional[Tensor] = None,
               tgt_padding_mask: Optional[Tensor] = None):
        """
        Autoregressive decoder
        """
        N, L = tgt.shape
        null_ctx = self.text_embed(tgt[:, :1])
        tgt_emb = self.pos_embed[:, :L-1] + self.text_embed(tgt[:, 1:])
        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))
        return self.at_decoder(tgt_emb, feature_map, tgt_mask, tgt_padding_mask)

    def iterative_decode(self, feature_map: torch.Tensor, input_labels=None):
        """
        Easy first decoding strategy
        :param feature_map:
        :param input_labels
        """
        bs = feature_map.shape[0]   # the batch size
        C = feature_map.shape[-1]

        tgt_tokens = torch.full((bs, self.max_len), self.mask_id, dtype=torch.long, device=self._device)
        pred_tgt_tokens = torch.full((bs, self.max_len), self.mask_id, dtype=torch.long, device=self._device)
        token_logits = torch.zeros([bs, self.max_len, len(self.tokenizer) - 2],
                                   dtype=torch.float, device=self._device)
        final_ffn = torch.zeros([bs, self.max_len, C], dtype=torch.float, device=self._device)

        for i in range(self.num_iter):
            # generate the corresponding mask and embed
            key_padding_mask = ((tgt_tokens == self.eos_id).cumsum(-1) > 0)
            mask_mask = (tgt_tokens == self.mask_id)
            key_padding_mask = key_padding_mask | mask_mask

            # checking for all true case
            for row in range(bs):
                if key_padding_mask[row, :].all():
                    key_padding_mask[row, :] = False

            predicts_embed = self.text_embed(tgt_tokens) + self.pos_embed
            outputs, alphas = self.decoder(predicts_embed, feature_map, key_padding_mask, attn_mask=None)
            new_token_logits = self.head(outputs).float()
            token_probs = new_token_logits.softmax(-1)
            new_ffn = outputs

            new_tgt_tokens = torch.argmax(new_token_logits, dim=-1)
            # only predict the mask ones
            token_probs = torch.max(token_probs, dim=-1).values     # N*T
            token_probs = torch.where(tgt_tokens == self.mask_id, token_probs, torch.zeros_like(token_probs))

            top_tuple = token_probs.topk(self.top_k, dim=1)     # get the top-k best position
            kth = torch.min(top_tuple.values, dim=1, keepdim=True).values
            update_idx = torch.greater_equal(token_probs, kth)

            logits_update_idx = torch.tile(update_idx.unsqueeze(dim=2), [1, 1, len(self.tokenizer) - 2])
            ffn_update_idx = torch.tile(update_idx.unsqueeze(dim=2), [1, 1, C])

            if input_labels is not None:    # is training
                tgt_tokens = torch.where(update_idx, input_labels, tgt_tokens)
                pred_tgt_tokens = torch.where(update_idx, new_tgt_tokens, pred_tgt_tokens)

            else:   # testing
                tgt_tokens = torch.where(update_idx, new_tgt_tokens, tgt_tokens)
                pred_tgt_tokens = tgt_tokens

            token_logits = torch.where(logits_update_idx, new_token_logits, token_logits)
            final_ffn = torch.where(ffn_update_idx, new_ffn, final_ffn)

        return token_logits, pred_tgt_tokens, alphas, final_ffn

    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        # encode the images
        feature_map = self.encode(images)
        # iterative_decoder
        iter_logits, _, _, _ = self.iterative_decode(feature_map)
        return iter_logits

    def forward_logits_loss(self, images: Tensor, labels: List[str]) -> Tuple[Tensor, Tensor, int]:
        bs = images.shape[0]
        targets = self.tokenizer.encode(labels, self._device, self.max_len)

        tgt_in = torch.full((bs, self.max_len), self.mask_id, dtype=torch.long, device=self._device)
        # encode the images
        feature_map = self.encode(images)
        # Decode using the parallel and autoregressive decoder
        logits, _, _, nat_glimpses = self.iterative_decode(feature_map)
        loss = F.cross_entropy(logits.flatten(end_dim=1), targets.flatten(), ignore_index=self.pad_id)
        loss_numel = (targets != self.pad_id).sum()
        return logits, loss, loss_numel

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        bs = images.shape[0]
        # generate the label's tokens
        tgt = self.tokenizer.encode(labels, self._device, self.max_len)
        # tgt = [l,a,b,l,e,[E],[P],[P]] bs * max_len
        # Encode the source sequence
        feature_map = self.encode(images)
        # Prepare the target sequences (input and output)
        tgt_in = tgt[:, :-1]    # bs * max_len-1
        tgt_in = torch.cat([torch.full((bs, 1), self.mask_id, dtype=torch.long, device=self._device), tgt_in], dim=1)

        # The [EOS] token is not depended upon by any other token in any permutation ordering
        tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)
        tgt_mask = torch.triu(torch.full((self.max_len, self.max_len), float('-inf'), device=self._device), 1)

        logits, _, _, nat_glimpses = self.iterative_decode(feature_map, input_labels=tgt)

        loss = F.cross_entropy(logits.flatten(end_dim=1), tgt.flatten(), ignore_index=self.pad_id)

        self.log('loss', loss)
        return loss
