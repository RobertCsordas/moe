import framework
import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data
import math
from typing import List, Tuple, Dict, Any
from models import TransformerLanguageModel
from ... import task, args
from layers.transformer import RelativeTransformerEncoderLayer, PrelnRelativeTransformerEncoderLayer
from layers.transformer.relative_preln_kvmem_transformer import PrelnRelativeKVMemTransformerEncoderLayer
from layers.transformer.relative_moe_transformer import RelativeMoeTransformerEncoderLayer
from layers.transformer.topk_transformer import TopkTransformer
from layers.moe_layer import MoE
from interfaces import Result


@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-lm.trafo.context_blocks", default=1)
    parser.add_argument("-lm.trafo.test_context_blocks", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-lm.trafo.test_pos_clamp", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-lm.trafo.same_length_eval", default=False)
    parser.add_argument("-lm.trafo.same_length", default=False)
    parser.add_argument("-lm.trafo.last_layer_context", default=False)
    parser.add_argument("-lm.trafo.xl_init", default=False)
    parser.add_argument("-lm.trafo.embedding_mode_init", default="default", choice=["default", "scale_to_sqrt_dmodel", "init_to_sqrt_dmodel", "one_and_scale_to_sqrt_dmodel", "like_preln"])
    parser.add_argument("-pkm.n_keys", default="128", parser=parser.int_list_parser)
    parser.add_argument("-pkm.n_heads", default=1)
    parser.add_argument("-pkm.knn", default=32)
    parser.add_argument("-pkm.stochastic", default=False)
    parser.add_argument("-pkm.query_batchnorm", default=False)
    parser.add_argument("-pkm.custom_init", default=0)
    parser.add_argument("-pkm.slice_values", default=False)
    parser.add_argument("-pkm.slice_proj", default=False)
    parser.add_argument("-pkm.sample_smallest", default=False)
    parser.add_argument("-moe.n_experts", default=128)
    parser.add_argument("-moe.expert_size", default=128)
    parser.add_argument("-moe.selection_mode", default="add", choice=["add", "gate", "sigmoid", "gumbel", "hard_gumbel", "predict", "predict_mlp", "classify", "gumbel_sigmoid", "sinkhorn", "sinkhorn2", "sinkmoid", "sinkmax", "moe", "mul", "random", "sinkmoid2", "sinkmax2", "modulate"])
    parser.add_argument("-moe.perplexity_reg", default=0.0)
    parser.add_argument("-moe.perplexity_reg_mode", default="step", choice=["step", "global", "time", "global_time"])
    parser.add_argument("-moe.reg_type", default="entropy", choice=["perplexity", "variance", "entropy", "l2", "switch", "normal"])
    parser.add_argument("-moe.key_mode", default="moe", choice=["moe", "both", "shared"])
    parser.add_argument("-moe.half_key", default=False)
    parser.add_argument("-moe.norm_keys", default=False)
    parser.add_argument("-moe.kmeans_distance", default='cosine', choice=['cosine', 'euclidean'])
    parser.add_argument("-moe.n_random", default=0)
    parser.add_argument("-moe.std_correction", default=False)
    parser.add_argument("-moe.topk_mode", default="full", choice=["full", "l1_approx", "approx"])
    parser.add_argument("-moe.activation_after_topk", default=False)
    parser.add_argument("-moe.weight_grouping", default="none", choice=["none", "keys_only", "keys_and_experts"])
    parser.add_argument("-moe.drop_parallel", default=True)
    parser.add_argument("-moe.mlp_selection", default=False)
    parser.add_argument("-moe.block_expert_sel_in_grad", default=False)
    parser.add_argument("-moe.classification_target", default="sum", choice=["sum", "max"])
    parser.add_argument("-moe.recluster_steps", default="", parser=parser.int_list_parser)
    parser.add_argument("-moe.norm_key_init", default=False)
    parser.add_argument("-moe.norm_value_init", default=False)
    parser.add_argument("-moe.norm_expert_sel_init", default=False)
    parser.add_argument("-moe.norm_standard_parallel_values", default=False)
    parser.add_argument("-moe.identical_init", default=False)
    parser.add_argument("-moe.topological_sel_reg", default=0.0)
    parser.add_argument("-moe.topological_expert_reg", default=0.0)
    parser.add_argument("-moe.sel_lr_multipler", default=1.0)
    parser.add_argument("-moe.expert_lr_multipler", default=1.0)
    parser.add_argument("-moe.gumbel_select_only", default=False)
    parser.add_argument("-moe.topk_value_norm_compensation", default=False)
    parser.add_argument("-moe.norm_expert_scores", default=False)
    parser.add_argument("-moe.sel_input_cluster_init", default=False)
    parser.add_argument("-moe.init_norm_mode", default="full")
    parser.add_argument("-moe.bias", default=False)
    parser.add_argument("-moe.sel_bias", default=False)
    parser.add_argument("-moe.rescale_normed", default=False)
    parser.add_argument("-moe.sel_norm", default="none", choice=["none", "cos", "input", "weights"])
    parser.add_argument("-moe.rescale_grads", default=False)
    parser.add_argument("-moe.gumbel_decay", default=0)
    parser.add_argument("-moe.sinkhorn_local", default=False)
    parser.add_argument("-moe.sinkhron_n_iters", default=3)
    parser.add_argument("-moe.dropout_factor", default=1.0)
    parser.add_argument("-moe.drop_expert", default=0.0)
    parser.add_argument("-moe.expert_size_init", default=False)
    parser.add_argument("-moe.sync_distributed", default=True)
    parser.add_argument("-moe.modulation_amplitude", default=0.5)
    parser.add_argument("-moe.invisible_selection", default=False)
    parser.add_argument("-moe.slope_multiplier", default=1.0)
    parser.add_argument("-moe.init_scale", default=1.0)
    parser.add_argument("-kvmem.linproj", default=False)
    parser.add_argument("-kvmem.head_merge_topk", default=False)
    parser.add_argument("-kvmem.load_balance", default=False)
    parser.add_argument("-kvmem.dropout", default="none", choice=["none", "early", "late", "weight", "score"])
    parser.add_argument("-kvmem.randomize_indices", default=False)
    parser.add_argument("-kvmem.standard_parallel", default=False)
    parser.add_argument("-kvmem.query_bias", default=False)
    parser.add_argument("-kvmem.approx_topk", default=False)
    parser.add_argument("-kvmem.norm_values", default=False)
    parser.add_argument("-kvmem.factorize", default=False)
    parser.add_argument("-kvmem.full_key", default=False)
    parser.add_argument("-kvmem.key_redundancy_factor", default=1)
    parser.add_argument("-kvmem.two_stage", default=False)
    parser.add_argument("-kvmem.head_exclusive", default=False)
    parser.add_argument("-transformer.topk_value", default=32)
    parser.add_argument("-transformer.universal.nonshared", default=0)
    parser.add_argument("-transformer.topk_use_norm", default=True)
    parser.add_argument("-transformer.activation", default="relu", choice=["relu", "topk", "gelu", "identity", "sigmoid", "softmax"])
    parser.add_argument("-transformer.p_drop_layer", default=0.0)
    parser.add_argument("-transformer.head_projection_size", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-transformer.ln_affine", default=True)
    parser.add_argument("-transformer.ln_after_attention", default=True)
    parser.add_argument("-transformer.output_mode", default="normal", choice=["normal", "sum", "geometric", "sigmoid"])


@task()
class TransformerLMMixin:
    helper: framework.helpers.TrainingHelper

    def is_preln(self) -> bool:
        return "preln" in self.helper.args.transformer.variant

    def topk_activation(self, x: torch.Tensor) -> torch.Tensor:
        nx = -x
        return torch.masked_fill(x, nx <= nx.kthvalue(self.helper.args.transformer.topk_value, keepdim=True)[0], 0)

    def get_layers(self) -> List[torch.nn.Module]:
        # pyright: reportOptionalMemberAccess=false
        if self.helper.args.transformer.activation == "relu":
            activation = F.relu
        elif self.helper.args.transformer.activation == "topk":
            activation = self.topk_activation
        elif self.helper.args.transformer.activation == "identity":
            activation = lambda x: x
        elif self.helper.args.transformer.activation == "sigmoid":
            activation = torch.sigmoid
        elif self.helper.args.transformer.activation == "gelu":
            activation = F.gelu
        elif self.helper.args.transformer.activation == "softmax":
            activation = lambda x: F.softmax(x, dim=-1)
        else:
            raise ValueError(f"Invalid activation: {self.helper.args.transformer.activation}")

        base_args = dict(
            d_model=self.helper.args.state_size,
            nhead=self.helper.args.transformer.n_heads,
            dim_feedforward=int(self.helper.args.state_size * self.helper.args.transformer.ff_multiplier),
            dropout=self.helper.args.dropout,
            activation=activation
        )


        extra_args = {} if not self.helper.args.transformer.variant.endswith("_gelu") else {
            "activation": F.gelu,
            "drop_expand": False
        }


        if self.helper.args.transformer.variant in {"preln_relative"}:
            mklayer = lambda: PrelnRelativeTransformerEncoderLayer(
                **base_args, **extra_args, test_pos_clamp=self.helper.args.lm.trafo.test_pos_clamp,
                n_layers=self.helper.args.transformer.encoder_n_layers,
                head_projection_size=self.helper.args.transformer.head_projection_size,)
        elif self.helper.args.transformer.variant in {"preln_topk"}:
            mklayer = lambda: TopkTransformer(
                **base_args, **extra_args, test_pos_clamp=self.helper.args.lm.trafo.test_pos_clamp,
                n_layers=self.helper.args.transformer.encoder_n_layers, k=self.helper.args.transformer.topk_value,
                use_norm=self.helper.args.transformer.topk_use_norm,
                head_projection_size=self.helper.args.transformer.head_projection_size,)
        elif self.helper.args.transformer.variant in {"preln_kvmem"}:
            mklayer = lambda: PrelnRelativeKVMemTransformerEncoderLayer(
                **base_args, **extra_args, test_pos_clamp=self.helper.args.lm.trafo.test_pos_clamp,
                n_layers=self.helper.args.transformer.encoder_n_layers, n_keys=self.helper.args.pkm.n_keys,
                pkm_stochastic=self.helper.args.pkm.stochastic, pkm_heads=self.helper.args.pkm.n_heads,
                pkm_custom_init=self.helper.args.pkm.custom_init, pkm_slice_values=self.helper.args.pkm.slice_values,
                pkm_knn=self.helper.args.pkm.knn, linproj=self.helper.args.kvmem.linproj,
                head_merge_topk=self.helper.args.kvmem.head_merge_topk,
                load_balance=self.helper.args.kvmem.load_balance, kvmem_dropout=self.helper.args.kvmem.dropout,
                kvmem_randomize_indices=self.helper.args.kvmem.randomize_indices,
                kvmem_query_bias=self.helper.args.kvmem.query_bias,
                standard_parallel=self.helper.args.kvmem.standard_parallel,
                approx_topk=self.helper.args.kvmem.approx_topk,
                factorize=self.helper.args.kvmem.factorize,
                full_key=self.helper.args.kvmem.full_key,
                key_redundancy_factor=self.helper.args.kvmem.key_redundancy_factor,
                two_stage=self.helper.args.kvmem.two_stage,
                head_exclusive=self.helper.args.kvmem.head_exclusive,
                head_projection_size=self.helper.args.transformer.head_projection_size,)
        elif self.helper.args.transformer.variant in {"preln_moe", "preln_moe_universal", "moe", "moe_universal"}:
            # def __init__(self, d_model, nhead, n_bins: int, bin_size: int, n_layers: int, dim_feedforward=2048,
            mklayer = lambda: RelativeMoeTransformerEncoderLayer(
                **base_args, **extra_args, preln=self.is_preln(),
                test_pos_clamp=self.helper.args.lm.trafo.test_pos_clamp,
                n_layers=self.helper.args.transformer.encoder_n_layers,
                standard_parallel=self.helper.args.kvmem.standard_parallel,
                custom_init=self.helper.args.pkm.custom_init,
                n_experts=self.helper.args.moe.n_experts,
                expert_size=self.helper.args.moe.expert_size,
                dropout_mode=self.helper.args.kvmem.dropout,
                knn=self.helper.args.pkm.knn,
                selection_mode=self.helper.args.moe.selection_mode,
                perplexity_reg=self.helper.args.moe.perplexity_reg,
                key_mode=self.helper.args.moe.key_mode,
                half_key=self.helper.args.moe.half_key,
                n_heads=self.helper.args.pkm.n_heads,
                norm_keys=self.helper.args.moe.norm_keys,
                perplexity_reg_mode=self.helper.args.moe.perplexity_reg_mode,
                n_random=self.helper.args.moe.n_random,
                reg_type=self.helper.args.moe.reg_type,
                std_correction=self.helper.args.moe.std_correction,
                topk_mode=self.helper.args.moe.topk_mode,
                head_projection_size=self.helper.args.transformer.head_projection_size,
                activation_after_topk=self.helper.args.moe.activation_after_topk,
                weight_grouping=self.helper.args.moe.weight_grouping,
                kmeans_distance=self.helper.args.moe.kmeans_distance,
                drop_parallel=self.helper.args.moe.drop_parallel,
                block_expert_sel_in_grad=self.helper.args.moe.block_expert_sel_in_grad,
                mlp_selection=self.helper.args.moe.mlp_selection,
                classification_target=self.helper.args.moe.classification_target,
                norm_key_init=self.helper.args.moe.norm_key_init,
                normalize_expert_sel_init=self.helper.args.moe.norm_expert_sel_init,
                norm_value_init=self.helper.args.moe.norm_value_init,
                norm_standard_parallel_values=self.helper.args.moe.norm_standard_parallel_values,
                identical_init=self.helper.args.moe.identical_init,
                topological_sel_reg=self.helper.args.moe.topological_sel_reg,
                topological_expert_reg=self.helper.args.moe.topological_expert_reg,
                gumbel_select_only=self.helper.args.moe.gumbel_select_only,
                topk_value_norm_compensation=self.helper.args.moe.topk_value_norm_compensation,
                norm_expert_scores=self.helper.args.moe.norm_expert_scores,
                sel_input_cluster_init=self.helper.args.moe.sel_input_cluster_init,
                init_norm_mode=self.helper.args.moe.init_norm_mode,
                sel_bias=self.helper.args.moe.sel_bias,
                bias=self.helper.args.moe.bias,
                rescale_normed=self.helper.args.moe.rescale_normed,
                sel_norm=self.helper.args.moe.sel_norm,
                rescale_grads=self.helper.args.moe.rescale_grads,
                gumbel_decay=self.helper.args.moe.gumbel_decay,
                ln_affine=self.helper.args.transformer.ln_affine,
                sinkhorn_local=self.helper.args.moe.sinkhorn_local,
                sinkhorn_n_iters=self.helper.args.moe.sinkhron_n_iters,
                moe_dropout_factor=self.helper.args.moe.dropout_factor,
                drop_expert=self.helper.args.moe.drop_expert,
                expert_size_init=self.helper.args.moe.expert_size_init,
                sync_distributed=self.helper.args.moe.sync_distributed,
                modulation_amplitude=self.helper.args.moe.modulation_amplitude,
                invisible_selection=self.helper.args.moe.invisible_selection,
                slope_multiplier=self.helper.args.moe.slope_multiplier,
                moe_init_scale=self.helper.args.moe.init_scale)
        else:
            assert False, "Invalid variant"

        layers = [mklayer() for _ in range(self.helper.args.transformer.encoder_n_layers)]
        return layers


    def fix_init(self, model):
        init_std = 0.02

        torch.nn.init.normal_(model.embedding.weight, 0.0, init_std)
        # torch.nn.init.normal_(model.embedding_adapter.weight, 0.0, init_std)

        initialized = 0
        for m in model.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)) and hasattr(m, "weight"):
                torch.nn.init.normal_(m.weight, 0.0, init_std)
                initialized += m.weight.numel()
            if isinstance(m, (torch.nn.Linear, torch.nn.LayerNorm)) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
                initialized += m.bias.numel()
            if isinstance(m, (torch.nn.LayerNorm)) and m.weight is not None:
                torch.nn.init.normal_(m.weight, 1.0, init_std)
                initialized += m.weight.numel()
            if isinstance(m, MoE):
                torch.nn.init.normal_(m.keys, 0.0, init_std)
                torch.nn.init.normal_(m.values, 0.0, init_std)
                if m.expert_sel is not None:
                    torch.nn.init.normal_(m.expert_sel, 0.0, init_std)
                    initialized += m.expert_sel.numel()
                initialized += m.keys.numel() + m.values.numel()

        print(f"Reinitialized {initialized/self.n_weights*100:.3f}% weights")

    def create_model(self) -> torch.nn.Module:
        # pyright: reportOptionalMemberAccess=false
        tlayers = self.get_layers()

        if self.helper.args.transformer.output_mode != "normal" and self.is_preln():
            raise ValueError("accumulated_output not supported with pre-ln")

        model = TransformerLanguageModel(
            len(self.train_set.vocabulary), self.helper.args.embedding_size,
            self.helper.args.state_size, self.helper.args.dropout,
            tied_embedding=self.helper.args.tied_embedding,
            layers=tlayers, n_prev_states=self.helper.args.lm.trafo.context_blocks,
            n_prev_states_test=self.helper.args.lm.trafo.test_context_blocks,
            same_length_eval=self.helper.args.lm.trafo.same_length_eval,
            p_drop_layer=self.helper.args.transformer.p_drop_layer,
            same_length=self.helper.args.lm.trafo.same_length,
            use_last_state=self.helper.args.lm.trafo.last_layer_context,
            norm_before_output=self.is_preln(), output_mode=self.helper.args.transformer.output_mode,)

        self.n_weights = sum(p.numel() for p in model.parameters())

        with torch.no_grad():
            if self.is_preln():
                model.embedding_scale = 1.0
            elif self.helper.args.lm.trafo.xl_init:
                self.fix_init(model)
            elif self.helper.args.lm.trafo.embedding_mode_init=="scale_to_sqrt_dmodel":
                norm = model.embedding.weight.norm(dim=-1).mean()
                model.embedding_scale = math.sqrt(self.helper.args.state_size) / norm
            elif self.helper.args.lm.trafo.embedding_mode_init=="one_and_scale_to_sqrt_dmodel":
                norm = model.embedding.weight.norm(dim=-1).mean()
                model.embedding_scale = math.sqrt(self.helper.args.state_size)
                model.embedding.weight.mul_(1.0 / norm)
            elif self.helper.args.lm.trafo.embedding_mode_init=="init_to_sqrt_dmodel":
                norm = model.embedding.weight.norm(dim=-1, keepdim=True)
                model.embedding_scale=1.0
                model.embedding.weight.mul_(math.sqrt(self.helper.args.state_size) / norm)

        return model

    def moe_recluster(self):
        for n, m in self.model.named_modules():
            if isinstance(m, MoE):
                perm = m.regroup_weights()
                m.patch_optimizer_state(self.optimizer, perm)

    def train_step(self) -> Tuple[Result, Dict[str, Any]]:
        if self.helper.args.kvmem.norm_values:
            with torch.no_grad():
                for m in self.model.modules():
                    if isinstance(m, torch.nn.EmbeddingBag):
                        m.weight.div_(m.weight.norm(dim=-1, keepdim=True))
        if self.helper.args.moe.recluster_steps:
            if self.helper.state.iter in self.helper.args.moe.recluster_steps:
                self.moe_recluster()

        return super().train_step()

    def get_optimizer_param_list(self):
        params = list(self.model.parameters())
        sel_params = []
        expert_params = []

        if self.helper.args.moe.sel_lr_multipler != 1.0:
            for m in self.model.modules():
                if isinstance(m, MoE):
                    sel_params += list(m.sel.parameters()) if m.mlp_selection else [m.expert_sel]

        if self.helper.args.moe.expert_lr_multipler != 1.0:
            for m in self.model.modules():
                if isinstance(m, MoE):
                    expert_params += [m.keys, m.values]

        excluded_params = [id(p) for p in sel_params + expert_params]
        params = [p for p in params if id(p) not in excluded_params]

        if not excluded_params:
            return params

        return [
            {"params": params},
            {"params": sel_params, "lr": self.helper.args.lr * self.helper.args.moe.sel_lr_multipler},
            {"params": expert_params, "lr": self.helper.args.lr * self.helper.args.moe.expert_lr_multipler},
        ]
