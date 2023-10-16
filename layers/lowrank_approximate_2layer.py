# Based on PKM implementation based on https://github.com/facebookresearch/XLM/blob/main/PKM-layer.ipynb

import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional, Sequence, List, Union
from dataclasses import dataclass
from layers import LoggingLayer
from framework import utils as U
from .transformer.transformer import ActivationFunction


@dataclass
class SparseAddress:
    scores: torch.Tensor
    indices: torch.LongTensor

    @classmethod
    def cat(cls, addresses: Sequence, dim: int):
        return SparseAddress(
            torch.cat([a.scores for a in addresses], dim),
            torch.cat([a.indices for a in addresses], dim),
        )

    @classmethod
    def stack(cls, addresses: Sequence, dim: int):
        return SparseAddress(
            torch.stack([a.scores for a in addresses], dim),
            torch.stack([a.indices for a in addresses], dim),
        )

    def view(self, *shape):
        return SparseAddress(
            self.scores.view(*shape),
            self.indices.view(*shape),
        )

    def expand_as(self, t):
        if isinstance(t, SparseAddress):
            t = t.scores
        return SparseAddress(
            self.scores.expand_as(t),
            self.indices.expand_as(t),
        )

    def __getitem__(self, *args):
        return SparseAddress(
            self.scores.__getitem__(*args),
            self.indices.__getitem__(*args)
        )

    def flatten(self, *args):
        return SparseAddress(
            self.scores.flatten(*args),
            self.indices.flatten(*args)
        )


class LowrankApproximate2Layer(LoggingLayer, torch.nn.Module):
    def __init__(self, n_dim: int, n_keys: Union[int, Tuple[int, int]], n_heads: int = 1, knn: int = 32,
                 dropout: float = 0, k_dim: Optional[int] = None, sparse: bool = False, stochastic: bool = False,
                 custom_init: int = 0, weight_scale: float = 1.0,
                 slice_values: bool = False, head_merge_topk: bool = False, load_balance: bool = False,
                 query_proj: bool = True, randomize_indices: bool = False, dropout_mode: str = "none",
                 query_bias: bool = False, approx: bool = False, factorize: bool = False, full_key: bool = False,
                 key_redundancy_factor: int = 1, two_stage: bool = False, factors: Optional[List[int]] = None,
                 head_exclusive: bool = False, activation: ActivationFunction = F.relu):

        super().__init__()

        # global parameters
        self.input_dim = n_dim
        self.output_dim = n_dim
        self.k_dim = k_dim or n_dim
        self.v_dim = n_dim
        n_keys = n_keys[0] if isinstance(n_keys, (tuple, list)) and len(n_keys) == 1 else n_keys
        self.key_sizes = [n_keys, n_keys] if isinstance(n_keys, int) else n_keys
        self.size = int(np.prod(self.key_sizes))
        self.heads = n_heads
        self.knn = knn
        self.stochastic = stochastic
        self.slice_values = slice_values
        self.head_merge_topk = head_merge_topk
        self.load_balance = load_balance
        self.dropout_mode = dropout_mode
        self.approx = approx
        self.factorize = factorize
        self.full_key = full_key
        self.two_stage = two_stage
        self.head_exclusive = head_exclusive
        self.custom_init = custom_init
        self.activation = activation

        self.no_knn = all([k <= self.knn for k in self.key_sizes]) or self.knn == 0

        if self.factorize:
            if factors is not None:
                if np.prod(factors) != self.knn:
                    raise ValueError("{factors} is not a factorization of {self.knn}")
                self.k = factors
            else:
                self.k = U.decompose_factors(self.knn, 2)
            print(f"Approximate2Layer: Using factorization: {self.k}")


        assert self.dropout_mode in ["none", "early", "late", "weight", "score"]

        assert self.k_dim >= 2 and self.k_dim % 2 == 0

        assert (not slice_values) or (self.v_dim % self.heads == 0), "Value dimension must be divisible by the num of heads."

        # dropout
        self.query_dropout = dropout

        if self.dropout_mode == "early":
            self.query_dropout = math.sqrt(self.query_dropout)

        # initialize keys / values
        self.real_vdim = (self.v_dim // n_heads) if slice_values else self.v_dim
        self.values = torch.nn.EmbeddingBag(self.size, self.real_vdim , mode='sum', sparse=sparse)

        self.keys = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty((self.heads, s * key_redundancy_factor, self.k_dim // (1 if self.full_key else 2)))) for s in self.key_sizes
        ])

        if self.two_stage:
            self.full_keys = torch.nn.Parameter(torch.empty((self.size, self.k_dim)))

        if self.head_exclusive:
            self.head_scales = torch.nn.Parameter(torch.zeros(self.size, self.heads+1))

        initializer = self.get_custom_init()
        for k in self.keys:
            initializer(k, std=n_dim ** -0.5 * weight_scale)

        if self.two_stage:
            initializer(self.full_keys, std=n_dim ** -0.5 * weight_scale)

        if custom_init in {0,1}:
            initializer(self.values.weight, std=n_dim ** -0.5 * weight_scale)
        elif custom_init in {2,4}:
            initializer(self.values.weight, std=(knn * self.heads) ** -0.5 * weight_scale)
        elif custom_init in {3,5}:
            initializer(self.values.weight, std=self.size ** -0.5 * weight_scale)
        else:
            raise ValueError(f"Invalid custom_init: {custom_init}")

        self.query_proj = torch.nn.Linear(n_dim, n_dim * n_heads, bias=query_bias) if query_proj else None

        self.register_buffer("usage_count", torch.zeros(self.size, dtype=torch.long), persistent=False)

        self.register_buffer("seq", torch.arange(self.knn, dtype=torch.long), persistent=False)
        self.register_buffer("head_shift", (torch.arange(self.heads, dtype=torch.float) * (self.key_sizes[0] / n_heads)).long(), persistent=False)

        self.log_count = 0
        self.randomize_indices = randomize_indices and self.heads > 1

    def get_custom_init(self):
        return torch.nn.init.normal_ if self.custom_init in {0, 4, 5} else U.init.trunc_normal_

    def topk(self, x: torch.Tensor, dim: int = -1, k: Optional[int] = None) -> SparseAddress:
        k = k or self.knn

        if k >= x.shape[dim] or k == 0:
            d = [1] * x.ndim
            d[dim] = -1
            return SparseAddress(
                x, torch.arange(x.shape[dim], device=x.device, dtype=torch.long).view(*d).expand_as(x))

        if self.approx:
            x = x.view(*x.shape[:-1], k, -1)
            scores, ind = x.max(-1)
            return SparseAddress(scores, self.seq[:k] * x.shape[-1] + ind)
        else:
            return SparseAddress(*x.contiguous().topk(k, dim=dim, sorted=False))

    def merge_sub_address(self, addr1: SparseAddress, addr2: SparseAddress) -> SparseAddress:
        # cartesian product on best candidate keys
        addr = self.combine_address_product(addr1, addr2)

        if self.dropout_mode == "late":
            addr.scores = F.dropout(addr.scores, p=self.query_dropout, training=self.training)

        # select best scores with associated indices
        addr2 = self.topk(addr.scores)
        addr2.indices = addr.indices.gather(-1, addr2.indices)

        return addr2

    def get_score(self, scores1: torch.Tensor, scores2: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        i1 = indices // self.key_sizes[-1]
        i2 = indices % self.key_sizes[-1]

        # return scores1[:, i1] + scores2[:, i2]
        return scores1.gather(-1, i1) + scores2.gather(-1, i2)

    @property
    def is_load_balane(self):
        return self.training and self.load_balance

    def index_combine(self, indices1: torch.Tensor, indices2: torch.Tensor) -> torch.Tensor:
        # Must be in sync with get_scores and get_dense_score
        return indices1 * self.key_sizes[-1] + indices2

    def combine_address_simple(self, addr1: SparseAddress, addr2: SparseAddress) -> SparseAddress:
        return SparseAddress(
            addr1.scores + addr2.scores,
            self.index_combine(addr1.indices, addr2.indices)
        )

    def combine_address_product(self, addr1: SparseAddress, addr2: SparseAddress) -> SparseAddress:
        return SparseAddress(
            (addr1.scores.unsqueeze(-1) + addr2.scores.unsqueeze(-2)).flatten(start_dim=-2),
            self.index_combine(addr1.indices.unsqueeze(-1), addr2.indices.unsqueeze(-2)).flatten(start_dim=-2)
        )

    def score_project(self, query: torch.Tensor, key1: torch.Tensor, key2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert query.dim() == 2 and query.size(1) == self.k_dim
        half = self.k_dim // 2

        # split query for product quantization
        if self.full_key:
            q1 = q2 = query
        else:
            q1 = query[:, :half]                                          # (bs,half)
            q2 = query[:, half:]                                          # (bs,half)

        # # compute indices with associated scores
        # if head_index % 2 == 1:
        #     q1, q2 = q2, q1

        if self.dropout_mode == "weight":
            key1 = F.dropout(key1, p=self.query_dropout, training=self.training)
            key2 = F.dropout(key2, p=self.query_dropout, training=self.training)

        scores1 = F.linear(q1, key1, bias=None)                 # (bs,n_keys)
        scores2 = F.linear(q2, key2, bias=None)                 # (bs,n_keys)

        if self.dropout_mode == "early":
            scores1 = F.dropout(scores1, p=self.query_dropout, training=self.training)
            scores2 = F.dropout(scores2, p=self.query_dropout, training=self.training)

        return scores1, scores2

    def _get_indices(self, query: torch.Tensor, key1: torch.Tensor, key2: torch.Tensor, head_index: int) -> SparseAddress:
        """
        Generate scores and indices for a specific head.
        """
        scores1, scores2 = self.score_project(query, key1, key2)

        if self.factorize:
            addr1 = self.topk(scores1, k=self.k[0])
            addr2 = self.topk(scores2, k=self.k[1])

            addr1.indices = addr1.indices % self.key_sizes[0]
            addr2.indices = addr2.indices % self.key_sizes[1]

            res = self.combine_address_product(addr1, addr2)
        elif not self.approx:
            with torch.no_grad():
                addr1 = self.topk(scores1)                                    # (bs,knn)
                addr2 = self.topk(scores2)                                    # (bs,knn)

                addr1.indices = addr1.indices % self.key_sizes[0]
                addr2.indices = addr2.indices % self.key_sizes[1]

                res = self.merge_sub_address(addr1, addr2)

            res.scores = self.get_score(scores1, scores2, res.indices)
        else:
            addr1 = self.topk(scores1)                                    # (bs,knn)
            addr2 = SparseAddress(*torch.max(scores2, -1, keepdim=True))

            # This order should be equivalent to the above but faster
            # addr1 = SparseAddress(*torch.max(scores1, -1, keepdim=True))
            # addr2 = self.topk(scores2)

            addr1.indices = addr1.indices % self.key_sizes[0]
            addr2.indices = addr2.indices % self.key_sizes[1]

            res = self.combine_address_simple(addr1, addr2)

        if self.head_exclusive:
            scale = torch.softmax(self.head_scales[res.indices], -1)[..., head_index]
            res.scores = res.scores * scale

        # assert (res2.scores == res.scores).all()
        # assert (res2.indices == res.indices).all()

        if self.is_load_balane:
            rind1 = torch.randint(0, self.key_sizes[0], addr1.indices.shape, dtype=addr1.indices.dtype, device=addr1.indices.device)
            rind2 = torch.randint(0, self.key_sizes[1], addr1.indices.shape, dtype=addr1.indices.dtype, device=addr1.indices.device)

            scores1_c = scores1.scatter(-1, addr1.indices, torch.zeros_like(addr1.indices, dtype=scores1.dtype))
            scores2_c = scores1.scatter(-1, addr2.indices, torch.zeros_like(addr2.indices, dtype=scores2.dtype))

            addr = SparseAddress(
                scores1_c.gather(-1, rind1) + scores2_c.gather(-1, rind2),
                self.index_combine(rind1, rind2)
            )

            res = SparseAddress.cat([res, addr], -1)

        return res

    def get_head_specific_queries(self, query: torch.Tensor) -> List[torch.Tensor]:
        if self.query_proj is not None:
            queries = query.view(-1, self.heads, self.k_dim).unbind(-2)
        else:
            query = query.view(-1, self.k_dim)
            queries = [query] * self.heads

        return queries

    def get_indices(self, query: torch.Tensor) -> SparseAddress:
        """
        Generate scores and indices.
        """

        queries = self.get_head_specific_queries(query)

        outputs = [self._get_indices(queries[i], self.keys[0][i], self.keys[1][i], i) for i in range(self.heads)]
        # for i in range(self.heads):
            # outputs[i].indices = (outputs[i].indices + (self.n_keys // self.heads) * i) % self.size

        if self.randomize_indices:
            for i in range(self.heads - 1):
                # outputs[i].indices = self.ind_perm[i][outputs[i].indices]
                outputs[i].indices = (outputs[i].indices + self.head_shift[i+1]) % self.size

        addr = SparseAddress.stack(outputs, -2)

        if self.head_merge_topk:
            addr2 = self.topk(addr.scores)
            addr2.indices = addr.indices.gather(-1, addr2.indices)
            addr = addr2

        return addr

    def get_dense_score(self, query: torch.Tensor, key1: torch.Tensor, key2: torch.Tensor) -> torch.Tensor:
        scores1, scores2 = self.score_project(query, key1, key2)
        return (scores1.unsqueeze(-1) + scores2.unsqueeze(-2)).flatten(-2)

    def get_indices_dense(self, query: torch.Tensor) -> torch.Tensor:
        queries = self.get_head_specific_queries(query)
        outputs = [self.get_dense_score(queries[i], self.keys[0][i], self.keys[1][i]) for i in range(self.heads)]
        return sum(outputs)

    def forward_dense(self, query: torch.Tensor) -> torch.Tensor:
        assert not self.two_stage

        scores = self.get_indices_dense(query)

        scores = self.activation(scores)                                         # (bs*heads,knn)
        if self.dropout_mode == "score":
            scores = F.dropout(scores, p=self.query_dropout, training=self.training)

        return F.linear(scores, self.values.weight.T)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Read from the memory.
        """
        # input dimensions
        assert input.shape[-1] == self.input_dim
        prefix_shape = input.shape[:-1]
        bs = np.prod(prefix_shape)

        # compute query
        query = self.query_proj(input) if self.query_proj is not None else input
        # query = query.contiguous().view(bs * self.heads, self.k_dim)            # (bs*heads,k_dim)

        if self.no_knn:
            return self.forward_dense(query).view_as(input)

        # retrieve indices and scores
        addr = self.get_indices(query)                               # (bs*heads,knn)

        # if self.head_exclusive:
        #     scales = torch.softmax(self.head_scales[addr.indices], -1)
        #     addr.scores = addr.scores * scales

        if not self.head_merge_topk:
            k = self.knn * (2 if self.is_load_balane else 1)
            addr = addr.view(-1, k) if self.slice_values else addr.view(bs, -1)

        # weighted sum of values
        if self.two_stage:
            real_keys = self.full_keys[addr.indices]
            addr.scores = torch.einsum("btd,bd->bt", real_keys, input.flatten(0, -2)) * torch.sigmoid(addr.scores)

        addr.scores = self.activation(addr.scores)                                        # (bs*heads,knn)
        if self.dropout_mode == "score":
            addr.scores = F.dropout(addr.scores, p=self.query_dropout, training=self.training)

        output = self.values(addr.indices, per_sample_weights=addr.scores.type_as(self.values.weight))  # (bs,v_dim)

        # if self.load_balance:
        #     self.usage_count.index_add_(0, addr.indices.flatten(), torch.ones(1, dtype=torch.long, device=addr.indices.device).expand(addr.indices.nelement()))

        if self.training:
            self.usage_count.index_add_(0, addr.indices.flatten(), torch.ones(1, dtype=torch.long, device=addr.indices.device).expand(addr.indices.nelement()))
            self.log_count += 1
            if self.log_count % 100 == 0:
                n_used = (self.usage_count > 0).long().sum()
                self.log("n_nonzero", n_used)
                self.log("n_zero", self.size - n_used)
                self.usage_count.fill_(0)

        # reshape output
        return output.view_as(input)                  # (...,v_dim)
