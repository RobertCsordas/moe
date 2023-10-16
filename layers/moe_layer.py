import torch
import torch.distributed
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, Tuple, List, Union, Optional
from layers import LoggingLayer
from layers import RegularizedLayer
from framework import utils
import framework
from framework.layers import gumbel_sigmoid_noise
import math
from layers import OncePerIterLayer
from layers import cvmm, cvmm_prepare_sel
from layers.cvmm.cvmm import CVMMSel
import os
import json


class MoE(LoggingLayer, RegularizedLayer, OncePerIterLayer, torch.nn.Module):
    def __init__(self, dmodel: int, n_experts: int, expert_size: int, n_heads: int, knn: int = 0,
                 dropout: float = 0, weight_scale: float = 1.0, custom_init: int = 0,
                 dropout_mode: str = "none", selection_mode: str = "add", perplexity_reg: float = 0.0,
                 key_mode: str = "moe", half_key: bool = False, norm_keys: bool = False,
                 perplexity_reg_mode: str="step", n_random: int = 0, reg_type: str = "entropy",
                 std_correction: bool = False, topk_mode: str = "full", activation_after_topk: bool = False,
                 weight_grouping: str = "none", kmeans_distance: str = "cosine",
                 activation = lambda x: F.relu(x, inplace=True), block_expert_sel_in_grad: bool = False,
                 mlp_selection: bool = False, classification_target: str = "sum",
                 normalize_expert_sel_init: bool = False, norm_key_init: bool = False, norm_value_init: bool = False,
                 identical_init: bool = False, topological_sel_reg: float = 0.0, topological_expert_reg: float = 0.0,
                 gumbel_select_only: bool = False, topk_value_norm_compensation: bool = False,
                 norm_expert_scores: bool = False, sel_input_cluster_init: bool = False,
                 n_parallel_expert_channels: int = 0, init_norm_mode: str = "full", sel_bias: bool = False,
                 bias: bool = False, rescale_normed: bool = False, sel_norm: str = "none",
                 rescale_grads: bool = False, gumbel_decay: int = 0, v_dim: Optional[int] = None,
                 sinkhorn_local: bool = False, sinkhorn_n_iters: int = 3, expert_dropout: float = 0.0,
                 expert_size_init: bool = False, sync_distributed: bool = False,
                 modulation_amplitude: float = 0.5, invisible_selection: bool = False,
                 slope_multiplier: float = 1.0):

        super().__init__()
        self.custom_init = custom_init
        self.k_dim = dmodel
        self.v_dim = v_dim if v_dim is not None else dmodel
        self.n_experts = n_experts
        self.expert_size = expert_size
        self.size = self.n_experts * self.expert_size
        self.knn = knn
        self.dropout = dropout
        self.dropout_mode = dropout_mode
        self.selection_mode = selection_mode
        self.perplexity_reg = perplexity_reg
        self.half_key = half_key
        self.key_mode = key_mode
        self.k_vec_dim = self.k_dim // (2 if half_key else 1)
        self.n_heads = n_heads
        self.norm_keys = norm_keys
        self.perplexity_reg_mode = perplexity_reg_mode
        self.n_random = n_random
        self.reg_type = reg_type
        self.topk_mode = topk_mode
        self.activation_after_topk = activation_after_topk
        self.weight_grouping = weight_grouping
        self.kmeans_distance = kmeans_distance
        self.activation = activation
        self.block_expert_sel_in_grad = block_expert_sel_in_grad
        self.mlp_selection = mlp_selection
        self.classification_target = classification_target
        self.weight_scale = weight_scale
        self.normalize_expert_sel_init = normalize_expert_sel_init
        self.norm_key_init = norm_key_init
        self.norm_value_init = norm_value_init
        self.identical_init = identical_init
        self.topological_sel_reg = topological_sel_reg
        self.topological_expert_reg = topological_expert_reg
        self.gumbel_select_only = gumbel_select_only
        self.topk_value_norm_compensation = topk_value_norm_compensation
        self.norm_expert_scores = norm_expert_scores
        self.sel_input_cluster_init = sel_input_cluster_init
        self.iter = 0
        self.layer = 0
        self.initalized = False
        self.rescale_normed = rescale_normed
        self.sel_norm = sel_norm
        self.rescale_grads = rescale_grads
        self.gumbel_decay = gumbel_decay
        self.was_training = True
        self.sinkhorn_local = sinkhorn_local
        self.sinkhorn_n_iters = sinkhorn_n_iters
        self.expert_dropout = expert_dropout
        self.reg_counts = 0
        self.sync_distributed = sync_distributed and torch.distributed.is_initialized()
        self.modulation_amplitude = modulation_amplitude
        self.invisible_selection = invisible_selection
        self.slope_multiplier = slope_multiplier

        self.coocurence = None

        assert self.selection_mode in {"add", "gate", "sigmoid", "gumbel", "hard_gumbel", "gumbel_sigmoid", "sinkhorn", "sinkhorn2", "sinkmoid", "sinkmax", "sinkhorn_local", "mul", "random", "sinkmoid2", "sinkmax2", "modulate"}
        assert self.perplexity_reg_mode in {"step", "global", "time", "global_time"}
        assert self.dropout_mode in {"none", "score"}
        assert self.reg_type in {"perplexity", "variance", "entropy", "l2", "switch"}
        assert self.topk_mode in {"full", "l1_approx", "approx"}
        assert self.weight_grouping in {"none", "keys_only", "keys_and_experts"}
        assert self.classification_target in {"sum", "max"}
        assert self.sel_norm in {"none", "cos", "input", "weights"}

        if selection_mode in {"mul"} and activation_after_topk:
            raise ValueError("Activation after topk is not supported with mul selection")

        if self.sel_norm != "none" and mlp_selection:
            raise ValueError("normalization not supported with mlp_selection")

        if std_correction and self.selection_mode in {"add"}:
            if key_mode == "both":
                self.key_std_correction = math.sqrt(3)
            else:
                self.key_std_correction = math.sqrt(2)
        elif std_correction and self.selection_mode in {"sigmoid", "sinkmoid", "sinkmoid2"}:
            self.key_std_correction = 2.0
        else:
            self.key_std_correction = 1.0

        if self.key_mode in {"moe", "both"}:
            self.keys = torch.nn.Parameter(torch.empty(self.n_experts, self.k_vec_dim, self.expert_size))
            self.get_initializer()(self.keys, std=dmodel ** -0.5 * weight_scale * self.key_std_correction)
        else:
            self.keys = None

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(self.n_experts, self.expert_size))
            self.o_bias = torch.nn.Parameter(torch.zeros(self.v_dim))
        else:
            self.bias = None
            self.o_bias = None

        if self.key_mode in {"shared", "both"}:
            self.shared_keys = torch.nn.Parameter(torch.empty(self.k_vec_dim, self.expert_size))
            self.get_initializer()(self.shared_keys, std=dmodel ** -0.5 * weight_scale * self.key_std_correction)
        else:
            self.shared_keys = None

        self.values = torch.nn.Parameter(torch.empty(self.n_experts, self.expert_size, self.v_dim))

        if self.mlp_selection:
            self.sel = torch.nn.Sequential(
                torch.nn.Linear(self.k_vec_dim, dmodel),
                torch.nn.ReLU(),
                torch.nn.Linear(dmodel, self.n_experts, bias=bias)
            )
            self.get_initializer()(self.sel[0].weight, std=self.k_vec_dim ** -0.5 * weight_scale * self.key_std_correction)
            self.get_initializer()(self.sel[-1].weight, std=dmodel ** -0.5 * weight_scale * self.key_std_correction)
            self.expert_sel = None
        else:
            self.sel = lambda x: F.linear(x, self.expert_sel, self.sel_bias)
            self.expert_sel = torch.nn.Parameter(torch.empty(self.n_experts, self.k_vec_dim))
            self.sel_bias = torch.nn.Parameter(torch.zeros(self.n_experts)) if sel_bias else None

            self.get_initializer()(self.expert_sel, std=self.k_vec_dim ** -0.5 * weight_scale)

        if init_norm_mode == "full":
            real_size = self.size
        elif init_norm_mode == "selected_experts":
            real_size = self.expert_size * self.n_heads
        elif init_norm_mode == "selected_channels":
            real_size = self.knn
        elif init_norm_mode == "expert_size":
            real_size = self.expert_size
        else:
            raise ValueError("Unknown init_norm_mode")

        real_size += n_parallel_expert_channels

        if expert_size_init:
            real_size = self.expert_size

        self.get_initializer()(self.values, std=real_size ** -0.5 * weight_scale)
        self.sel_hist = []
        self.index_sel_counts = 0
        self.index_sel_norm = 0

        self.index_sel_counts_100 = 0
        self.index_sel_norm_100 = 0

        self.sel_count_log = None

        self.register_buffer("kv_sel_counts", torch.zeros(self.n_experts, self.expert_size), persistent=False)
        self.register_buffer("kv_sel_counts_100", torch.zeros_like(self.kv_sel_counts))

        if self.rescale_normed and self.sel_norm != "none":
            self.sel_scale = torch.nn.Parameter(torch.ones([1]))
        else:
            self.sel_scale = 1.0

        if self.norm_expert_scores:
            self.expert_scale = torch.nn.Parameter(torch.full([1], math.sqrt(expert_size)))

        self.register_buffer("seq", torch.arange(max(self.knn, self.n_heads, self.n_experts, self.k_dim, self.v_dim), dtype=torch.long), persistent=False)
        self.regroup_weights()

    def keys_to_logical_order(self, keys: torch.Tensor) -> torch.Tensor:
        k = keys.view(self.n_experts, self.k_vec_dim, self.expert_size)
        return k.permute(0, 2, 1).contiguous().view(-1, self.k_vec_dim)

    def keys_from_logical_order(self, keys: torch.Tensor) -> torch.Tensor:
        return keys.view(self.n_experts, self.expert_size, self.k_vec_dim).permute(0, 2, 1).contiguous().view(self.n_experts * self.k_vec_dim, self.expert_size)

    def init_sel(self, x: torch.Tensor):
        if not self.sel_input_cluster_init:
            return

        with torch.no_grad():
            from kmeans_pytorch import kmeans
            _, cluster_centers = kmeans(
                X=x, num_clusters=self.n_experts, distance=self.kmeans_distance, device=torch.device('cuda')
            )

            self.expert_sel.set_(cluster_centers.to(self.expert_sel.device).contiguous())
            if self.normalize_expert_sel_init:
                self.renorm_keep_std(self.expert_sel, dim=1)

    def renorm_keep_std(self, weight: torch.Tensor, dim: int = 0):
        with torch.no_grad():
            std = weight.std()
            weight.div_(weight.norm(dim=dim, keepdim=True))
            weight.mul_(std / weight.std())

    def regroup_weights(self) -> Optional[torch.Tensor]:
        with torch.no_grad():

            if self.norm_key_init:
                self.renorm_keep_std(self.keys.view(self.n_experts, self.k_vec_dim, self.expert_size), dim=1)

            if self.norm_value_init:
                self.renorm_keep_std(self.values, dim=1)

            if self.identical_init:
                k = self.keys.view(self.n_experts, self.k_vec_dim, self.expert_size)
                self.keys.set_(k[:1].expand_as(k).reshape_as(self.keys))

                v = self.values.view(self.n_experts, self.expert_size, self.v_dim)
                self.values.set_(v[:1].expand_as(v).reshape_as(self.values))

            ids = None
            if self.weight_grouping != "none":
                #  self.n_experts * self.k_vec_dim, self.expert_size
                k = self.keys_to_logical_order(self.keys)

                from kmeans_pytorch import kmeans
                cluster_ids_x, cluster_centers = kmeans(
                    X=k, num_clusters=self.n_experts, distance=self.kmeans_distance, device=torch.device('cuda')
                )

                _, ids = cluster_ids_x.sort()
                k = self.keys_from_logical_order(k[ids])

                self.keys.set_(k.contiguous())
                self.values.set_(self.values[ids].contiguous())
                if self.weight_grouping == "keys_and_experts":
                    self.expert_sel.set_(cluster_centers.contiguous().to(self.expert_sel.device))
                else:
                    self.get_initializer()(self.expert_sel, std=self.k_vec_dim ** -0.5 * self.weight_scale)

            if self.normalize_expert_sel_init:
                self.renorm_keep_std(self.expert_sel, dim=1)

            return ids

    def patch_optimizer_state(self, optimizer: torch.optim.AdamW, ids: torch.Tensor):
        if self.weight_grouping == "none":
            return

        with torch.no_grad():
            ks = optimizer.state[self.keys]
            vs = optimizer.state[self.values]

            for p in {"exp_avg", "exp_avg_sq"}:
                k = self.keys_to_logical_order(ks[p])
                ks[p].set_(self.keys_from_logical_order(k[ids]))

                vs[p].set_(vs[p][ids])

            es = optimizer.state[self.expert_sel]
            for p in {"exp_avg", "exp_avg_sq", 'step'}:
                es[p].zero_()

    def get_initializer(self):
        return torch.nn.init.normal_ if self.custom_init in {0} else utils.init.trunc_normal_

    def sparse_matmul(self, indices: torch.Tensor, values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return F.embedding_bag(indices, weight.type_as(values), per_sample_weights=values, mode="sum", sparse=False)

    # def sparse_matmul(self, indices: torch.Tensor, values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    #     sin = torch.sparse_csr_tensor(
    #         crow_indices=torch.arange(0, values.nelement() + 1, values.shape[-1], device=indices.device),
    #         col_indices=indices.flatten(),
    #         values=values.flatten(),
    #         size=(values.shape[0], weight.shape[0])
    #     )
    #     return sin @ weight.type_as(values)

    def pre_train_forward(self):
        if self.norm_keys:
            with torch.no_grad():
                self.keys.div_(self.keys.norm(dim=-1, keepdim=True))

        if self.topk_value_norm_compensation:
            with torch.no_grad():
                self.value_norms = self.values.norm(2, dim=-1)

    def topoloss(self, x: torch.Tensor) -> torch.Tensor:
        return (F.mse_loss(x[1:], x[:-1], reduction='mean') +
                F.mse_loss(x[1:], x[:-1], reduction='mean'))

    def ani(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2
        chunk_size = 32

        xnorm = F.normalize(x, 2, dim=-1)

        accu = 0
        for i in range(0, x.shape[0], chunk_size):
            a = xnorm[i: i + chunk_size]
            sims = xnorm @ a.T
            sims[i : i + chunk_size].fill_diagonal_(0)
            accu += sims.sum()

        return accu / (x.shape[0] * (x.shape[0] - 1))

    def log_expert_sel_usage(self, prefix: str, channel_sel_counts: torch.Tensor):
        sel_nonzero = (channel_sel_counts != 0).type(torch.float).sum(axis=-1) / self.expert_size
        self.log(f"{prefix}/mean", sel_nonzero.mean())
        self.log(f"{prefix}/min", sel_nonzero.min())
        self.log(f"{prefix}/max", sel_nonzero.max())


    def post_train_forward(self):
        if self.training and self.rescale_grads:
            self.values.grad.view(self.n_experts, -1).mul_(self.rescale[:, None])
            self.keys.grad.view(self.n_experts, -1).mul_(self.rescale[:, None])
            self.expert_sel.grad.mul_(self.rescale[:, None])

    def pre_train_forward(self):
        if self.training and not self.was_training:
            sorted_counts = self.index_sel_counts.sort(descending=True).values
            self.log("test_exert_channel_usage", framework.visualize.plot.Barplot(sorted_counts, xlabel="expert", ylabel="usage count"), drop_old=True)

        self.layer = 0
        if self.sel_hist:
            self.sel_hist = []
        self.index_sel_counts = 0
        self.index_sel_norm = 0
        self.reg_counts = 0

    def before_loss(self):
        if self.sel_hist:
            # Concatenate against time dimension. Important for the within-batch regularization
            sel = torch.cat(self.sel_hist, -2)
            self.add_perplexity_reg(sel)

            self.sel_hist = []

        if self.topological_sel_reg > 0:
            self.add_reg(lambda: self.topological_sel_reg * self.topoloss(self.expert_sel))

        if self.topological_expert_reg > 0:
            self.add_reg(lambda: self.topological_expert_reg * (
                self.topoloss(self.keys.view(self.n_experts, -1)) +
                self.topoloss(self.values.view(self.n_experts, -1))
            ))

        if self.rescale_grads:
            self.rescale = 1.0 / self.index_sel_counts.clamp(min=1)

        # json.dumps


        if self.index_sel_norm > 0:
            if self.training:
                with torch.no_grad():
                    self.log("usag_rel_perplexity_all_layers", utils.relative_perplexity(self.index_sel_counts / self.index_sel_norm))
                    self.log("dead_expert_proportion_all_layers", (self.index_sel_counts == 0).float().sum() / self.n_experts)

                    self.log_expert_sel_usage("exert_channel_usage", self.kv_sel_counts)

                    self.kv_sel_counts_100.add_(self.kv_sel_counts)
                    self.kv_sel_counts.zero_()

                    self.index_sel_counts_100 = self.index_sel_counts_100 + self.index_sel_counts
                    self.index_sel_norm_100 = self.index_sel_norm_100 + self.index_sel_norm

                    if self.training and self.iter % 100 == 0:
                        norm_cnt = self.index_sel_counts_100 / self.index_sel_norm_100
                        self.log("usag_rel_perplexity_100", utils.relative_perplexity(norm_cnt))
                        self.log("dead_expert_proportion_100", (self.index_sel_counts_100 == 0).float().sum() / self.n_experts)

                        sorted_counts = self.index_sel_counts_100.sort(descending=True).values
                        self.log("usage_counts_100", framework.visualize.plot.Barplot(sorted_counts, xlabel="expert", ylabel="usage count"), drop_old=True)


                        self.log_expert_sel_usage("exert_channel_usage_100", self.kv_sel_counts_100)
                        self.kv_sel_counts_100.zero_()

                        self.index_sel_counts_100 = 0
                        self.index_sel_norm_100 = 0

                        self.log("ani/keys", self.ani(self.keys_to_logical_order(self.keys)))
                        self.log("ani/values", self.ani(self.values.flatten(0, -2)))
                        self.log("ani/expert_sel", self.ani(self.expert_sel.T))

        if self.training:
            self.iter += 1

    def topk(self, x: torch.Tensor, k: int, approx: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if approx:
            x = x.view(*x.shape[:-1], k, -1)
            scores, ind = x.max(-1)
            return scores, self.seq[:k] * x.shape[-1] + ind
        else:
            return x.topk(k, dim=-1, sorted=False)

    def add_perplexity_reg(self, sel: torch.Tensor):
        sync_distributed = self.sync_distributed and (self.perplexity_reg_mode not in {"time", "global_time"})

        def log_mean(x: torch.Tensor, dim: int = 0):
            if sync_distributed:
                xlse = framework.utils.distributed_ops.logsumexp(x, dim=dim)

                # Normalize
                n = torch.tensor(x.shape[dim]).to(x.device)
                torch.distributed.all_reduce(n, op=torch.distributed.ReduceOp.SUM)
                return xlse - n.log()
            else:
                return x.logsumexp(dim) - math.log(x.shape[dim])

        if self.perplexity_reg_mode in {"time", "global_time"}:
            sel = sel.flatten(0, -3)
        else:
            sel = sel.flatten(0, -2)

        # Note: sel are raw logits, no matter what activation is used
        if self.perplexity_reg > 0:
            if self.reg_type == "perplexity":
                sel_d = F.log_softmax(sel, dim=-1)
                sel_d = log_mean(sel_d, -2)
                loss = lambda: self.perplexity_reg * ( - utils.relative_perplexity_l(sel_d).mean())
            elif self.reg_type == "entropy":
                sel_d = F.log_softmax(sel, dim=-1)
                sel_d = log_mean(sel_d, -2)
                loss = lambda: self.perplexity_reg * ( - utils.entropy_l(sel_d).mean())
            elif self.reg_type == "variance":
                if sync_distributed:
                    raise NotImplementedError("Variance regularization is not supported in distributed mode")
                avg_sel = sel.mean(-2)
                loss = lambda: self.perplexity_reg * avg_sel.var(-1).mean()
            elif self.reg_type == "l2":
                loss = lambda: self.perplexity_reg * sel.pow(2).mean()
            elif self.reg_type == "switch":
                if sync_distributed:
                    torch.distributed.all_reduce(self.reg_counts, op=torch.distributed.ReduceOp.SUM)

                p_sel_real = self.reg_counts / self.reg_counts.sum(-1, keepdims=True)
                if self.perplexity_reg_mode in {"time", "global_time"}:
                    p_sel_real = p_sel_real.unsqueeze(-2)

                loss = lambda: self.perplexity_reg * (F.softmax(sel, dim=-1) * p_sel_real).mean()
                self.reg_counts = 0
            else:
                assert False

            self.add_reg(loss, "moe")

    def compute_scores(self, input: torch.Tensor, index: CVMMSel, expert_scores: torch.Tensor, shared_score: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.keys is not None:
            # scores = self.sparse_matmul(
            #     (self.seq[:input.shape[-1]] + index[:, None] * (self.k_dim // (2 if self.half_key else 1))),
            #     input,
            #     self.keys
            # )
            scores = cvmm(input, index, self.keys)
            if self.shared_keys is not None:
                scores = scores + shared_score
        else:
            scores = shared_score

        if self.bias is not None:
            scores = scores + self.bias[index.raw_sel]

        if self.invisible_selection:
            unmodulated_scores = scores
            scores = scores.detach()

        if self.selection_mode in {"add"}:
            with torch.no_grad():
                self.log("expert_key_positive_rate", (scores > 0).type_as(scores).mean())
            scores = scores + expert_scores[..., None]
        elif self.selection_mode in {"mul"}:
            scores = scores * expert_scores[..., None]
        elif self.selection_mode in {"gate", "sigmoid", "gumbel", "gumbel_sigmoid", "sinkhorn", "sinkhorn2", "sinkmoid", "sinkmax", "random", "modulate", "sinkmoid2"}:
            # Handle it later
            pass
        elif self.selection_mode == "hard_gumbel":
            s = (torch.ones_like(expert_scores) - expert_scores).detach() + expert_scores
            scores = scores * s[..., None]

        if self.invisible_selection and scores is not unmodulated_scores:
            scores = unmodulated_scores + scores - scores.detach()

        scores = self.activation(scores)

        if self.norm_expert_scores:
            scores = F.normalize(scores, 1, dim=-1) * self.expert_scale

        if self.selection_mode in {"gate", "sigmoid", "gumbel", "gumbel_sigmoid", "sinkhorn", "sinkhorn2", "sinkmoid", "sinkmax", "modulate", "sinkmoid2"}:
            if self.invisible_selection:
                unmodulated_scores = scores
                scores = scores.detach()
            scores = scores * expert_scores[..., None]
            if self.invisible_selection:
                scores = unmodulated_scores + scores - scores.detach()

        if self.train and self.iter % 10 == 0:
            with torch.no_grad():
                gt0 = (scores > 0).float()
                gt0_s = gt0.sum()
                if self.selection_mode in {"add"}:
                    self.log("k1_vs_k2_magnitude", (scores / expert_scores[..., None]).sum() / gt0_s - 1)

                self.log("relu_pass_rate", gt0_s / scores.numel())

                self.kv_sel_counts.index_add_(0, index.raw_sel.flatten(), gt0.flatten(end_dim=-2))


        # elif self.selection_mode in {"predict_rank"}:
        #     self.add_reg(lambda: self.rank_loss(expert_scores, scores.detach().sum(-1)))

        if self.dropout > 0 and self.dropout_mode != "none":
            scores = F.dropout(scores, self.dropout, training=self.training)

        # indices = torch.arange(0, scores.shape[-1], device=input.device) + index[:, None] * self.expert_size
        return scores

    def sel_activation(self, sel: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        reg_sel = sel
        if self.selection_mode in {"gumbel", "hard_gumbel"}:
            if self.training:
                sel = F.gumbel_softmax(sel)
            else:
                sel = F.softmax(sel)
        elif self.selection_mode == "gumbel_sigmoid":
            if self.training and (self.gumbel_decay == 0 or self.gumbel_decay > self.iter):
                noise = gumbel_sigmoid_noise(sel)
                if self.gumbel_decay:
                    noise = noise * (1 - self.iter / self.gumbel_decay)
                sel = sel + noise
            else:
                sel = F.sigmoid(sel)
        elif self.selection_mode in {"sinkhorn", "sinkmoid", "sinkmax"}:
            if self.training:
                if self.sinkhorn_local:
                    sel = sel.view(-1, seq_len, sel.shape[-1])

                for _ in range(self.sinkhorn_n_iters):
                    if self.sinkhorn_local or (not self.sync_distributed):
                        sel = sel - torch.logsumexp(sel, -2, keepdim=True)
                    else:
                        sel = sel - framework.utils.distributed_ops.logsumexp(sel, -2, keepdim=True)

                    sel = sel - torch.logsumexp(sel, -1, keepdim=True)
                reg_sel = sel

                if self.sinkhorn_local:
                    sel = sel.flatten(end_dim=-2).exp()

                sel = sel.exp()
            elif self.selection_mode == "sinkmoid":
                sel = F.sigmoid(sel)
            else:
                sel = F.softmax(sel, dim=-1)
        elif self.selection_mode in {"sinkhorn2", "sinkmoid2", "sinkmax2"}:
            if self.training:
                sel = self.sinkhorn(sel, self.selection_mode != "sinkmoid2")
            elif self.selection_mode == "sinkmoid":
                sel = F.sigmoid(sel)
            else:
                sel = F.softmax(sel, dim=-1)
        elif self.selection_mode in {"sigmoid"}:
            sel = torch.sigmoid(sel)
        elif self.selection_mode in {"modulate"}:
            sel = torch.tanh(sel) * (self.modulation_amplitude / 0.5) + 1
        elif self.selection_mode in {"add"}:
            sel = sel
        elif self.selection_mode in {"mul"}:
            sel = sel.abs()
            reg_sel = sel
        elif self.selection_mode in {"gate"}:
            sel = F.softmax(sel, dim=-1)
            with torch.no_grad():
                self.log("expert_rel_perplexity_per_selection", utils.relative_perplexity(sel).mean())
        else:
            assert False

        return sel, reg_sel

    def sinkhorn(self, x: torch.Tensor, normalize:bool = True) -> torch.Tensor:
        # Based on
        A, B = x.shape[-2:]

        a = torch.zeros_like(x[..., 0, :])
        b = torch.zeros_like(x[..., 0])

        for _ in range(self.sinkhorn_n_iters):
            b = math.log(A) - (x - a[..., None, :]).logsumexp(-1)
            if self.sync_distributed:
                a = math.log(B) - framework.utils.distributed_ops.logsumexp(x - b[..., None], -2)
            else:
                a = math.log(B) - (x - b[..., None]).logsumexp(-2)

        r = (a[..., None, :] + b[..., None] + x).exp()

        if normalize and self.sync_distributed:
            A = torch.tensor(A, device=x.device)
            A = torch.distributed.reduce_all(A, op=torch.distributed.ReduceOp.SUM)
            A = A.item()
        return (r / (A * B)) if normalize else r

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.initalized:
            self.init_sel(input)
            self.initalized = True

        out = 0

        if self.half_key:
            in1 = input[..., :self.k_dim // 2]
            in2 = input[..., self.k_dim // 2:]
        else:
            in1 = in2 = input

        if self.selection_mode != "random":
            if self.block_expert_sel_in_grad:
                in1 = in1.detach()

            sel = self.sel(in1) * self.slope_multiplier

            if self.sel_norm == "cos":
                sel = sel / (in1.norm(dim=-1, keepdim=True) * self.expert_sel.norm(dim=-1)[None]) * self.sel_scale
            elif self.sel_norm == "weights":
                sel = sel * (self.sel_scale / self.expert_sel.norm(dim=-1)[None])
            elif self.sel_norm == "input":
                sel = sel * (self.sel_scale / in1.norm(dim=-1, keepdim=True))

            sel_raw = reg_sel = sel

            inv_val = float("-inf")

            if (not self.activation_after_topk) or self.selection_mode in {"sinkhorn", "sinkhorn2", "gumbel", "hard_gumbel", "gumbel_sigmoid", "sinkmoid", "sinkmax", "mul", "sinkmoid2"}:
                # Sinkhorn should be always applied before top-k
                sel, reg_sel = self.sel_activation(sel, input.shape[-2])
                if self.selection_mode not in {"sinkmoid", "sinkmoid2"}:
                    inv_val = 0

            if self.training and self.expert_dropout > 0:
                if self.selection_mode not in {"sigmoid", "modulate", "gate", "sinkmoid", "sinkmoid2"}:
                    raise ValueError("Expert dropout not supported in this mode")

                mask = torch.rand_like(sel) < self.expert_dropout
                sel2 = sel.masked_fill(mask, inv_val)
            else:
                sel2 = sel

            sel_val, sel_index = self.topk(sel2, self.n_heads, self.topk_mode in {"l1_approx", "approx"})

            if self.activation_after_topk or (self.selection_mode in {"sinkmoid", "sinkmax", "mul", "sinkmoid2"}) or (self.gumbel_select_only and self.selection_mode in {"gumbel", "hard_gumbel", "gumbel_sigmoid", "gumbel_sigmoid", "sinkmax"}):
                sel_val = torch.gather(sel_raw, -1, sel_index)
                if self.selection_mode in {"gumbel_sigmoid", "sinkmoid", "sinkmoid2"}:
                    sel_val = torch.sigmoid(sel_val)
                elif self.selection_mode in {"sinkhorn", "sinkhorn2"}:
                    # In case of sinkhorn, simulate the effect of post-topk activation by renormalizing
                    sel_val = F.normalize(sel_val, p=1, dim=-1)
                else:
                    sel_val, reg_sel = self.sel_activation(sel_val, input.shape[-2])
        else:
            sel_index = torch.randint(0, self.n_experts, (*input.shape[:-1], self.n_heads), device=input.device)
            sel_val = torch.ones_like(sel_index, dtype=input.dtype, device=input.device)
            reg_sel = None


        record_counts_now = (self.training and self.iter % 10 == 0) or (not self.training)

        if not self.training:
            sel_index_flat = sel_index.flatten(end_dim=-2)
            if self.coocurence is None:
                self.coocurence = torch.zeros([self.n_experts, self.n_experts], device=sel_index_flat.device, dtype=torch.long)

            for h1 in range(self.n_heads):
                for h2 in range(self.n_heads):
                    ind_flat = sel_index_flat[..., h1] * self.n_experts + sel_index_flat[..., h2]
                    values = torch.tensor([1], device=self.coocurence.device, dtype=self.coocurence.dtype).expand_as(ind_flat)
                    # values = sel_val[..., h2].flatten()
                    self.coocurence.flatten().put_(ind_flat, values, accumulate=True)
                    # self.coocurence[sel_index_flat[..., h1], sel_index_flat[..., h2]] += 1

        if record_counts_now or self.reg_type == "switch":
            reg_counts = F.one_hot(sel_index, self.n_experts).type_as(input)

        if self.reg_type == "switch":
            reg_counts2 = reg_counts.view(*input.shape[:-2], input.shape[-2] * self.n_heads, self.n_experts)
            if self.perplexity_reg_mode == "time":
                reg_counts2 = reg_counts2.sum(-2)
            else:
                reg_counts2 = reg_counts2.flatten(end_dim=-2).sum(0)

            self.reg_counts = self.reg_counts + reg_counts2

        if record_counts_now:
            with torch.no_grad():
                sel_counts = reg_counts.flatten(end_dim=-2).sum(0)
                cnt = sel_index.nelement()

                p_expert_sel = sel_counts / cnt

                self.index_sel_counts = self.index_sel_counts + sel_counts
                self.index_sel_norm = self.index_sel_norm + cnt

                if self.training:
                    self.log("min_sel_score", sel_val.min(dim=-1).values.mean())
                    self.log("max_sel_score", sel_val.max(dim=-1).values.mean())

                    sel_oh = F.one_hot(sel_index, self.n_experts).sum(-2).bool()
                    if self.layer >= 1 and self.training:
                        self.log(f"layer_sel_overlap_{self.layer}", ((self.prev_sel_oh & sel_oh).sum(-1).float() / self.n_heads).mean())

                    self.prev_sel_oh = sel_oh

                    ppl = utils.relative_perplexity(p_expert_sel)
                    self.log("usage_rel_perplexity", ppl)
                    self.log("dead_expert_proportion", (p_expert_sel == 0).float().sum() / self.n_experts)

        if self.perplexity_reg_mode in {"step", "time"}:
            self.add_perplexity_reg(reg_sel)
        elif self.perplexity_reg > 0 and self.training:
            self.sel_hist.append(reg_sel)

        shared_score = (in2 @ self.shared_keys) if self.shared_keys is not None else None

        scores_l = []

        sel_indices = [cvmm_prepare_sel(sel_index[..., h].int(), self.n_experts) for h in range(sel_index.shape[-1])]

        for h in range(sel_index.shape[-1]):
            hi = sel_indices[h]

            scores = self.compute_scores(in2, hi, sel_val[..., h], shared_score)
            scores_l.append(scores)

        if self.knn > 0 or self.selection_mode == "classify":
            with torch.no_grad():
                scores = torch.cat(scores_l, -1)

        if self.knn > 0:
            with torch.no_grad():
                tresh = scores.kthvalue(scores.shape[-1] - self.knn, -1).values

            scores_l = [s.masked_fill_(s < tresh[:, None], 0) for s in scores_l]

        out = 0
        for (hi, scores) in zip(sel_indices, scores_l):
            out = out + cvmm(scores, hi, self.values)

        # indices = torch.cat(ind_l, dim=-1)
        # scores = torch.cat(scores_l, dim=-1)

        if self.selection_mode == "classify":
            self.add_reg(lambda: self.cls_loss(sel_val, scores))

        # if self.knn > 0:
        #     if self.topk_value_norm_compensation:
        #         norms = self.value_norms[None].expand(indices.shape[0], -1).gather(-1, indices)
        #         scores2 = scores * norms
        #         _, ind2 = self.topk(scores2, self.knn, self.topk_mode == "approx")
        #         indices = indices.gather(-1, ind2)
        #         scores = scores.gather(-1, ind2)
        #     else:
        #         scores, ind2 = self.topk(scores, self.knn, self.topk_mode == "approx")
        #         indices = indices.gather(-1, ind2)

        # if self.n_random > 0 and self.selection_mode not in {"predict", "classify"}:
        #     with torch.no_grad():
        #         rind = torch.arange(0, self.n_experts, device=input.device)
        #         rind = torch.masked_select(rind, ~F.one_hot(sel_index, self.n_experts).sum(-2).bool()).view(in_flat.shape[0],-1)
        #         rind = rind.gather(-1, torch.randint(0, rind.shape[-1], size=[*rind.shape[:-1], self.n_random], device=rind.device))

        #     ind_l = [indices]
        #     scores_l = [scores]
        #     for i in range(self.n_random):
        #         hi = rind[..., i]
        #         indices, scores = self.compute_scores(in2, hi, sel.gather(-1, hi[:, None]).squeeze(), shared_score)

        #         ind_l.append(indices)
        #         scores_l.append(scores)

        #     indices = torch.cat(ind_l, dim=-1)
        #     scores = torch.cat(scores_l, dim=-1)

        # out = self.sparse_matmul(indices, scores, self.values)

        self.layer += 1

        self.was_training = self.training
        res = out.view(*input.shape[:-1], self.v_dim)
        if self.o_bias is not None:
            res = res + self.o_bias
        return res

    def dump_logs(self, save_dir: str):
        if self.coocurence is not None:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(self.coocurence, os.path.join(save_dir, "coocurence.pt"))

    def get_logs(self) -> Dict[str, Any]:
        res = super().get_logs()

        if self.coocurence is not None:
            coo = self.coocurence / self.coocurence.diagonal().clamp(min=1)[:, None]
            res["expert_coocurence"] = framework.visualize.plot.Heatmap(coo, xlabel="expert", ylabel="expert", textval=False)
            self.coocurence = None
        return res
