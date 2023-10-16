import torch
import torch.distributed
import torch.nn.functional as F
from typing import Tuple, List, Optional
import math
from .cvmm import cvmm, cvmm_prepare_sel, CVMMSel


def dist_logsumexp(x: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    # Calculate numerically stable distributed logsumexp
    xmax = x.max(dim=dim, keepdim=True).values
    torch.distributed.all_reduce(xmax, op=torch.distributed.ReduceOp.MAX)

    xe = (x - xmax).exp().sum(dim=dim, keepdim=True)
    torch.distributed.all_reduce(xe, op=torch.distributed.ReduceOp.SUM)

    res = (xmax + xe.log())
    if not keepdim:
        res = res.squeeze(dim)

    return res


def log_mean(x: torch.Tensor, dim: int = 0):
    if torch.distributed.is_initialized():
        xlse = dist_logsumexp(x, dim=dim)

        # Normalize
        n = torch.tensor(x.shape[dim]).to(x.device)
        torch.distributed.all_reduce(n, op=torch.distributed.ReduceOp.SUM)
        return xlse - n.log()
    else:
        return x.logsumexp(dim) - math.log(x.shape[dim])


def entropy_l(l: torch.Tensor) -> torch.Tensor:
    return - (l * l.exp()).sum(-1)


class MoE(torch.nn.Module):
    def __init__(self, dmodel: int, n_experts: int, expert_size: int, n_heads: int,
                 dropout: float = 0, selection_mode: str = "sigmoid",
                 activation_after_topk: bool = False,
                 activation=F.relu,
                 bias: bool = False, v_dim: Optional[int] = None,
                 sinkhorn_n_iters: int = 3, expert_dropout: float = 0.0,
                 weight_std_scale: float = 1.0):

        super().__init__()
        self.k_dim = dmodel
        self.v_dim = v_dim if v_dim is not None else dmodel
        self.n_experts = n_experts
        self.expert_size = expert_size
        self.size = self.n_experts * self.expert_size
        self.dropout = dropout
        self.selection_mode = selection_mode
        self.k_vec_dim = self.k_dim
        self.n_heads = n_heads
        self.activation_after_topk = activation_after_topk
        self.activation = activation
        self.sinkhorn_n_iters = sinkhorn_n_iters
        self.expert_dropout = expert_dropout

        if self.selection_mode not in {"softmax", "sigmoid", "sinkmoid"}:
            raise ValueError(f"Unknown selection mode {self.selection_mode}")

        self.keys = torch.nn.Parameter(torch.empty(self.n_experts, self.k_vec_dim, self.expert_size))
        self.values = torch.nn.Parameter(torch.empty(self.n_experts, self.expert_size, self.v_dim))
        self.expert_sel = torch.nn.Parameter(torch.empty(self.n_experts, self.k_vec_dim))
        torch.nn.init.normal_(self.keys, std=dmodel ** -0.5 * weight_std_scale)
        torch.nn.init.normal_(self.values, std=self.size ** -0.5 * weight_std_scale)
        torch.nn.init.normal_(self.expert_sel, std=self.k_vec_dim ** -0.5 * weight_std_scale)

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(self.n_experts, self.expert_size))
            self.o_bias = torch.nn.Parameter(torch.zeros(self.v_dim))
        else:
            self.bias = None
            self.o_bias = None

        self.renorm_keep_std(self.expert_sel, dim=1)

    def renorm_keep_std(self, weight: torch.Tensor, dim: int = 0):
        with torch.no_grad():
            std = weight.std()
            weight.div_(weight.norm(dim=dim, keepdim=True))
            weight.mul_(std / weight.std())

    def entropy_reg(self, sel: torch.Tensor) -> float:
        # Everything is done in log scale
        sel = sel.flatten(0, -2)
        sel = F.log_softmax(sel, dim=-1)
        sel = log_mean(sel, -2)
        return - entropy_l(sel).mean()

    def compute_scores(self, input: torch.Tensor, index: CVMMSel, expert_scores: torch.Tensor) -> torch.Tensor:
        scores = cvmm(input, index, self.keys)
        if self.bias is not None:
            scores = scores + self.bias[index.raw_sel]

        scores = self.activation(scores)
        scores = scores * expert_scores[..., None]

        if self.dropout > 0:
            # Standard dropout on the "up-projected scores"
            scores = F.dropout(scores, self.dropout, training=self.training)

        return scores

    def sel_activation(self, sel: torch.Tensor) -> torch.Tensor:
        if self.selection_mode == "sinkmoid":
            if self.training:
                with torch.no_grad():
                    sel = self.sinkhorn_unnorm(sel, False)
            else:
                sel = torch.sigmoid(sel)
        elif self.selection_mode == "sigmoid":
            sel = torch.sigmoid(sel)
        elif self.selection_mode == "softmax":
            sel = F.softmax(sel, dim=-1)
        else:
            assert False

        return sel

    def sinkhorn_unnorm(self, x: torch.Tensor) -> torch.Tensor:
        # Based on https://arxiv.org/abs/2202.01169. Unnormalized verison
        A, B = x.shape[-2:]

        a = torch.zeros_like(x[..., 0, :])
        b = torch.zeros_like(x[..., 0])

        for _ in range(self.sinkhorn_n_iters):
            b = math.log(A) - (x - a[..., None, :]).logsumexp(-1)
            if torch.distributed.is_initialized():
                a = math.log(B) - dist_logsumexp(x - b[..., None], -2)
            else:
                a = math.log(B) - (x - b[..., None]).logsumexp(-2)

        return (a[..., None, :] + b[..., None] + x).exp()

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Selection score calculation
        sel = sel_raw = F.linear(input, self.expert_sel, None)
        reg_loss = self.entropy_reg(sel_raw)

        # Selection activation and topk
        if (not self.activation_after_topk) or (self.selection_mode == "sinkmoid"):
            # Sinkhorn should be always applied before top-k
            sel = self.sel_activation(sel)

        if self.training and self.expert_dropout > 0:
            mask = torch.rand_like(sel) < self.expert_dropout
            sel = sel.masked_fill(mask, float("-inf"))

        sel_val, sel_index = sel.topk(self.n_heads, dim=-1, sorted=False)

        if self.activation_after_topk or (self.selection_mode == "sinkmoid"):
            sel_val = torch.gather(sel_raw, -1, sel_index)
            # for sinkmoid, the score is always calculated by a sigmoid
            sel_val = torch.sigmoid(sel_val) if self.selection_mode == "sinkmoid" else self.sel_activation(sel_val)

        # Preprocess the selection indices. They will be needed for both layers and save some time
        sel_indices = [cvmm_prepare_sel(sel_index[..., h].int(), self.n_experts) for h in range(sel_index.shape[-1])]

        # "Up-projection" layer for each head
        scores_l = [self.compute_scores(input, sel_indices[h], sel_val[..., h]) for h in range(sel_index.shape[-1])]

        # Down projection layer for each head
        out = 0
        for (hi, scores) in zip(sel_indices, scores_l):
            out = out + cvmm(scores, hi, self.values)

        res = out.view(*input.shape[:-1], self.v_dim)
        if self.o_bias is not None:
            res = res + self.o_bias
        return res, reg_loss
