import torch
from typing import Optional
from .relative_preln_transformer import PrelnRelativeTransformerEncoderLayer, AttentionMask, ActivationFunction
import torch.nn.functional as F
from layers import LoggingLayer


class TopkTransformer(PrelnRelativeTransformerEncoderLayer, LoggingLayer):
    def __init__(self, d_model, nhead, n_layers: int, dim_feedforward=2048, dropout=0.1,
                 activation: ActivationFunction = F.relu, attention_dropout=0,
                 test_pos_clamp: Optional[int] = None, drop_expand: bool = True, k: int = 32,
                 use_norm: bool = True, head_projection_size: Optional[int] = None):

        super().__init__(d_model, nhead, n_layers, dim_feedforward, dropout, activation, attention_dropout,
                         test_pos_clamp, drop_expand, head_projection_size=head_projection_size)

        LoggingLayer.__init__(self)
        self.k = k
        self.use_norm = use_norm

    def forward(self, src: torch.Tensor, mask: Optional[AttentionMask] = None, attend_to: Optional[torch.Tensor] = None,
                pos_offset: Optional[int] = None) -> torch.Tensor:
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, self.norm1(attend_to) if attend_to is not None else src2, mask,
                              pos_offset=pos_offset)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)

        middle = self.dropout(self.activation(self.linear1(src2)))

        with torch.no_grad():
            if self.use_norm:
                norms = self.linear2.weight.norm(dim=0)
                vals = - middle * norms
            else:
                vals = - middle
            mask = vals > vals.kthvalue(self.k, keepdim=True)[0]

        self.log("relu_pass_rate_before", (middle > 0).float().mean())

        middle = middle.masked_fill(mask, 0)

        self.log("topk_positive_rate", (middle > 0).float().sum(-1).mean()/self.k)

        src2 = self.linear2(middle)
        src = src + self.dropout2(src2)
        return src
