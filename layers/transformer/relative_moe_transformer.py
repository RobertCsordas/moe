
from typing import Optional, List, Union, Tuple
import torch
import torch.nn
import torch.nn.functional as F
from .transformer import ActivationFunction
from .multi_head_relative_pos_attention import FixedRelativeMultiheadAttention, AttentionMask
from .transformer_preln import reset_prenorm_params
from layers.moe_layer import MoE
import math
from framework import utils
from torch.profiler import profile, record_function, ProfilerActivity
from layers import LoggingLayer


class RelativeMoeTransformerEncoderLayer(LoggingLayer, torch.nn.Module):
    def __init__(self, d_model, nhead, n_experts: int, expert_size: int, n_layers: int, dim_feedforward=2048,
                 dropout=0.1, activation: ActivationFunction = F.relu, attention_dropout=0,
                 test_pos_clamp: Optional[int] = None, knn: int = 0,
                 standard_parallel: bool = False, custom_init: int = 0,
                 dropout_mode: str = "none", selection_mode: str = "add",
                 perplexity_reg: float = 0.0, key_mode: str = "moe", half_key: bool = False,
                 n_heads: int = 1, norm_keys: bool = False, perplexity_reg_mode: str="step",
                 n_random: int = 0, reg_type: str = "normal", std_correction: bool = False,
                 topk_mode: str = "full", head_projection_size: Optional[int] = None,
                 activation_after_topk: bool = False, weight_grouping: str = "none",
                 kmeans_distance: str = "cosine", drop_parallel: bool = True, block_expert_sel_in_grad: bool = False,
                 mlp_selection: bool = False, classification_target: str = "sum",
                 normalize_expert_sel_init: bool = False, norm_key_init: bool = False, norm_value_init: bool = False,
                 norm_standard_parallel_values: bool = False, identical_init: bool = False,
                 topological_sel_reg: float = 0.0, topological_expert_reg: float = 0.0,
                 gumbel_select_only: bool = False, topk_value_norm_compensation: bool = False,
                 norm_expert_scores: bool = False, sel_input_cluster_init: bool = False,
                 init_norm_mode: str = "full", sel_bias: bool = False,
                 bias: bool = False, rescale_normed: bool = False, sel_norm: str = "none",
                 rescale_grads: bool = False, gumbel_decay: int = 0, preln: bool = True, ln_affine: bool = True,
                 sinkhorn_local: bool = False, sinkhorn_n_iters: int = 3, moe_dropout_factor: float = 1.0,
                 drop_expert: float = 0.0, expert_size_init: bool = False, sync_distributed: bool = True,
                 modulation_amplitude: float = 0.5, invisible_selection: bool = False,
                 slope_multiplier: float = 1.0, moe_init_scale: float = 1.0):
        super().__init__()
        self.preln = preln
        self.i = 0
        self.self_attn = FixedRelativeMultiheadAttention(
            d_model, nhead, dropout=attention_dropout, test_pos_clamp=test_pos_clamp,
            projection_size=head_projection_size)

        std_scale = math.sqrt(2.0 / n_layers) if preln else 1.0
        std_scale *= math.sqrt(moe_init_scale)

        self.pkm = MoE(
            d_model, n_experts, expert_size, knn=knn, dropout=dropout * moe_dropout_factor, dropout_mode=dropout_mode,
            weight_scale=std_scale, custom_init=custom_init, selection_mode=selection_mode,
            perplexity_reg=perplexity_reg, key_mode=key_mode, half_key=half_key, n_heads=n_heads,
            norm_keys=norm_keys, perplexity_reg_mode=perplexity_reg_mode, n_random=n_random,
            reg_type=reg_type, std_correction=std_correction, topk_mode=topk_mode,
            activation_after_topk=activation_after_topk, weight_grouping=weight_grouping,
            kmeans_distance=kmeans_distance, activation=activation, block_expert_sel_in_grad=block_expert_sel_in_grad,
            mlp_selection=mlp_selection, classification_target=classification_target,
            normalize_expert_sel_init=normalize_expert_sel_init, norm_key_init=norm_key_init,
            norm_value_init=norm_value_init, identical_init=identical_init, topological_sel_reg=topological_sel_reg,
            topological_expert_reg=topological_expert_reg, gumbel_select_only=gumbel_select_only,
            topk_value_norm_compensation=topk_value_norm_compensation, norm_expert_scores=norm_expert_scores,
            sel_input_cluster_init=sel_input_cluster_init,
            n_parallel_expert_channels=dim_feedforward if standard_parallel else 0,
            init_norm_mode=init_norm_mode, sel_bias=sel_bias, bias=bias, rescale_normed=rescale_normed,
            sel_norm=sel_norm, rescale_grads=rescale_grads, gumbel_decay=gumbel_decay,
            sinkhorn_local=sinkhorn_local, sinkhorn_n_iters=sinkhorn_n_iters, expert_dropout=drop_expert,
            expert_size_init=expert_size_init, sync_distributed=sync_distributed,
            modulation_amplitude=modulation_amplitude, invisible_selection=invisible_selection,
            slope_multiplier=slope_multiplier)

        self.norm1 = torch.nn.LayerNorm(d_model, elementwise_affine=ln_affine)
        self.norm2 = torch.nn.LayerNorm(d_model, elementwise_affine=ln_affine)
        self.dropout = torch.nn.Dropout(dropout)

        self.activation = activation
        self.standard_parallel = standard_parallel
        self.drop_parallel = drop_parallel

        if preln:
            reset_prenorm_params(self, n_layers)

        if self.standard_parallel:
            self.linear1 = torch.nn.Linear(d_model, dim_feedforward, bias=bias)
            self.linear2 = torch.nn.Linear(dim_feedforward, d_model, bias=False)

            s_real = dim_feedforward + self.pkm.size
            # s_real = dim_feedforward + self.pkm.heads * self.pkm.knn

            init = self.pkm.get_initializer()

            init(self.linear1.weight, std=std_scale * math.sqrt(1.0 / d_model))
            init(self.linear2.weight, std=std_scale * math.sqrt(1.0 / s_real))

            if norm_standard_parallel_values:
                with torch.no_grad():
                    self.linear2.weight.div_(self.linear2.weight.norm(dim=0, keepdim=True))


    def forward(self, src: torch.Tensor, mask: Optional[AttentionMask] = None, attend_to: Optional[torch.Tensor] = None,
                pos_offset: Optional[int] = None) -> torch.Tensor:

        src2 = self.norm1(src) if self.preln else src
        src2 = self.self_attn(src2, self.norm1(attend_to) if attend_to is not None else src2, mask,
                              pos_offset=pos_offset)
        src = src + self.dropout(src2)

        if self.preln:
            src2 = self.norm2(src)
        else:
            src = src2 = self.norm1(src)

        if self.i == 3:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                src3 = self.pkm(src2)
            prof.export_chrome_trace("trace.json")
            assert False
        else:
            src3 = self.pkm(src2)

        # self.i += 1

        if self.standard_parallel:
            x = self.linear1(src2)
            with torch.no_grad():
                self.log("standard_parallel_relu_pass_rate", (x > 0).flatten(end_dim=-2).float().mean().item())
            x = self.activation(x)
            if self.drop_parallel:
                x = self.dropout(x)
            src3 = src3 + self.linear2(x)

        src = src + self.dropout(src3)
        if not self.preln:
            src = self.norm2(src)

        return src
