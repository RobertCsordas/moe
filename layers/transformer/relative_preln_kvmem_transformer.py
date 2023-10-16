
from typing import Optional, List, Union, Tuple
import torch
import torch.nn
import torch.nn.functional as F
from .transformer import ActivationFunction
from .multi_head_relative_pos_attention import FixedRelativeMultiheadAttention, AttentionMask
from .transformer_preln import reset_prenorm_params
from layers.lowrank_approximate_2layer import LowrankApproximate2Layer
import math


class PrelnRelativeKVMemTransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, n_keys: Union[int, Tuple[int, int]], n_layers: int, dim_feedforward=2048,
                 dropout=0.1, activation: ActivationFunction = F.relu, attention_dropout=0,
                 test_pos_clamp: Optional[int] = None, pkm_heads: int = 1, pkm_stochastic: bool = True,
                 pkm_custom_init: int = 0, pkm_slice_values: bool = False,
                 pkm_knn: int = 32, linproj: bool = False, head_merge_topk: bool = False, load_balance: bool = True,
                 kvmem_dropout: str = "none", kvmem_randomize_indices: bool = False, kvmem_query_bias: bool = False,
                 standard_parallel: bool = False, approx_topk: bool = False, factorize: bool = False,
                 full_key: bool = False, key_redundancy_factor: int = 1, two_stage: bool = False,
                 factors: Optional[List[int]] = None, head_exclusive: bool = False,
                 head_projection_size: Optional[int] = None):
        super().__init__()
        self.self_attn = FixedRelativeMultiheadAttention(
            d_model, nhead, dropout=attention_dropout, test_pos_clamp=test_pos_clamp,
            projection_size=head_projection_size)

        self.pkm = LowrankApproximate2Layer(
            d_model, n_keys, pkm_heads, stochastic=pkm_stochastic, custom_init=pkm_custom_init,
            weight_scale=math.sqrt(2.0 / n_layers), slice_values=pkm_slice_values, knn=pkm_knn,
            head_merge_topk=head_merge_topk, load_balance=load_balance, dropout=dropout,
            query_proj=linproj, randomize_indices=kvmem_randomize_indices, dropout_mode=kvmem_dropout,
            query_bias=kvmem_query_bias, approx=approx_topk, factorize=factorize, full_key=full_key,
            key_redundancy_factor=key_redundancy_factor, two_stage=two_stage, factors=factors,
            head_exclusive=head_exclusive, activation=activation)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)

        self.activation = activation
        self.standard_parallel = standard_parallel

        reset_prenorm_params(self, n_layers)

        if self.standard_parallel:
            self.linear1 = torch.nn.Linear(d_model, dim_feedforward, bias=False)
            self.linear2 = torch.nn.Linear(dim_feedforward, d_model, bias=False)

            initializer = self.pkm.get_custom_init()

            s_real = dim_feedforward + self.pkm.size
            # s_real = dim_feedforward + self.pkm.heads * self.pkm.knn
            initializer(self.linear2.weight, std=math.sqrt(2 / (n_layers * s_real)))
            initializer(self.pkm.values.weight, std=math.sqrt(2 / (n_layers * s_real)))
            initializer(self.linear1.weight, std=math.sqrt(2 / (n_layers * d_model)))

            if self.pkm.two_stage:
                initializer(self.pkm.full_keys, std=math.sqrt(2 / (n_layers * d_model)))


    def forward(self, src: torch.Tensor, mask: Optional[AttentionMask] = None, attend_to: Optional[torch.Tensor] = None,
                pos_offset: Optional[int] = None) -> torch.Tensor:
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, self.norm1(attend_to) if attend_to is not None else src2, mask,
                              pos_offset=pos_offset)
        src = src + self.dropout(src2)
        src2 = self.norm2(src)
        src3 = self.pkm(src2)

        if self.standard_parallel:
            src3 = src3 + self.linear2(self.dropout(self.activation(self.linear1(src2))))

        src = src + self.dropout(src3)
        return src
