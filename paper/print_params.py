import lib
from run_tests import get_runs_and_infos, dataset, n_test_blocks, test_field_name
from run_tests import get_runs_and_infos
from collections import OrderedDict
import math


runs, infos = get_runs_and_infos(["wikitext103_small_match_manyexperts", "wikitext103_baseline_big", "wikitext103_moe_big_drop_expert", "enwik8_moe_unshared_xl_nonshared_schedule_lowreg_exp_drop", "enwik8_baseline", "wikitext103_small_moe_unshared_xl_nonshared_schedule_short_128", "wikitext103_small_xl_nonshared_schedule_short_matched", "wikitext103_moe_smallbig"])

datasets = [
    ("Wikitext 103", {"task": "wikitext103_sp_transformer"}),
    ("Enwik8", {"task": "enwik8_transformer"}),
]

def simple_param(name):
    def get(config):
        return config[name]
    return get

params = OrderedDict()
params["$\\dmodel$"] = simple_param("state_size")
params["$\\dff$"] = lambda c: int(c["state_size"] * c["transformer.ff_multiplier"])
params["$\\nlayers$"] = simple_param("transformer.encoder_n_layers")
params["$\\nheads$"] = simple_param("transformer.n_heads")
params["head size"] = lambda c: c["transformer.head_projection_size"] if c["transformer.head_projection_size"]!="none" else (math.ceil(c["state_size"] / c["transformer.n_heads"]))
params["context size"] = simple_param("lm.unroll")
params["batch size"] = simple_param("batch_size")
params["dropout"] = simple_param("dropout")
params["lr warmup"] = lambda c: c["lr_warmup"] if c["lr_warmup"] > 0 else "-"


print("\\begin{tabular}{lr" + ("r" * len(params)) + "}")
print("\\toprule")
print("Dataset & \\#params & " + " & ".join(params.keys()) + " \\\\")
print("\\midrule")

for ds, f in datasets:
    f2 = {"transformer.variant": "preln_relative"}
    f2.update(f)
    runs2 = [r for r in runs if all(r.config.get(k) == v for k, v in f2.items())]
    i = sorted(range(len(runs2)), key=lambda i: infos[runs2[i].id]["n_param"])
    runs2 = [runs2[i] for i in i]

    for r in runs2:
        np = round(infos[r.id]["n_param"] / 1e6)
        pvalues = [k(r.config) for k in params.values()]
        pvalues = [str(p) for p in pvalues]
        print(f"{ds} & {np}M & " + " & ".join(pvalues) + " \\\\")

print("\\bottomrule")
print("\\end{tabular}")


moe_params = OrderedDict()
moe_params["$\\dmodel$"] = simple_param("state_size")
moe_params["$N_E$"] = simple_param("moe.n_experts")
moe_params["$G$"] = simple_param("moe.expert_size")
moe_params["$K$"] = simple_param("pkm.n_heads")
moe_params["$\\delta$"] = lambda c: c["moe.drop_expert"] if c["moe.drop_expert"] > 0 else "-"
moe_params["$\\gamma$"] = lambda c: c["moe.perplexity_reg"] if c["moe.perplexity_reg"] > 0 else "-"

print("\\begin{tabular}{lr" + ("r" * len(moe_params)) + "}")
print("\\toprule")
print("Dataset & \\#params & " + " & ".join(moe_params.keys()) + " \\\\")
print("\\midrule")

for ds, f in datasets:
    f2 = {"transformer.variant": "preln_moe"}
    f2.update(f)
    runs2 = [r for r in runs if all(r.config.get(k) == v for k, v in f2.items())]
    i = sorted(range(len(runs2)), key=lambda i: infos[runs2[i].id]["n_param"])
    runs2 = [runs2[i] for i in i]

    for r in runs2:
        np = round(infos[r.id]["n_param"] / 1e6)
        pvalues = [k(r.config) for k in moe_params.values()]
        pvalues = [str(p) for p in pvalues]
        print(f"{ds} & {np}M & " + " & ".join(pvalues) + " \\\\")

print("\\bottomrule")
print("\\end{tabular}")