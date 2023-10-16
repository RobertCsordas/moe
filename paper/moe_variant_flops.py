from collections import OrderedDict
from run_tests import get_runs_and_infos, dataset, n_test_blocks, test_field_name

runs, infos = get_runs_and_infos([
    "wikitext103_small_moe_unshared_xl_nonshared_schedule_short_128_lowreg",
    "wikitext103_small_moe_unshared_xl_nonshared_schedule_short_128_lowreg_softmax",
    "wikitext103_small_moe_unshared_xl_nonshared_schedule_short_128_lowreg_softmax_pre",
    "wikitext103_small_moe_unshared_xl_nonshared_schedule_short_128_lowreg_bad_init",
    "wikitext103_small_moe_unshared_xl_nonshared_schedule_short_128_lowreg_k",
    "wikitext103_small_moe_unshared_xl_nonshared_schedule_short_128_lowreg_es64",
    "wikitext103_small_moe_unshared_xl_nonshared_schedule_short_128_lowreg_k1_big",
    "wikitext103_small_moe_unshared_xl_nonshared_schedule_short_128_noreg",
    "wikitext103_small_moe_unshared_xl_nonshared_schedule_short_128_noreg_exp_drop",
    "wikitext103_small_moe_unshared_xl_nonshared_schedule_short_128_lowreg_dropout",
    "wikitext103_small_moe_unshared_xl_nonshared_schedule_short_128_lowreg_sinkmoid2",
    'wikitext103_small_moe_unshared_xl_nonshared_schedule_short_128_lowreg_switch_h1_small_fix_size_reg_init',
    "wikitext103_small_moe_unshared_xl_nonshared_schedule_short_128_lowreg_switch_h1_big_fix_size_reg_init",
    "wikitext103_small_moe_unshared_xl_nonshared_schedule_short_128_lowreg_switch_h4_fix_reg",
    "wikitext103_small_moe_unshared_xl_nonshared_schedule_short_128_lowreg_sinkmoid2_h1",
    "wikitext103_small_moe_unshared_xl_nonshared_schedule_short_128_lowreg_k8_64",



    "enwik8_moe_unshared_xl_nonshared_schedule_lowreg_exp_drop",
    "enwik8_moe_unshared_xl_nonshared_schedule_lowreg_exp_drop_softmax",
    "enwik8_moe_unshared_xl_nonshared_schedule_lowreg_exp_drop_bad_init",
    "enwik8_moe_unshared_xl_nonshared_schedule_lowreg_exp_drop_k",
    "enwik8_moe_unshared_xl_nonshared_schedule_lowreg_dropout",
    "enwik8_moe_unshared_xl_nonshared_schedule_noreg",
    "enwik8_moe_unshared_xl_nonshared_schedule_lowreg_exp_drop_k1_es512",
    "enwik8_moe_unshared_xl_nonshared_schedule_lowreg_exp_drop_sinkmoid2",
    "enwik8_moe_unshared_xl_nonshared_schedule_lowreg_exp_switch_es512_fixreg_init",
    "enwik8_moe_unshared_xl_nonshared_schedule_lowreg_exp_switch_es128_fixreg_init",
    "enwik8_moe_unshared_xl_nonshared_schedule_lowreg_exp_switch_h4",
    "enwik8_moe_unshared_xl_nonshared_schedule_lowreg_exp_drop_sinkmoid2_h1",
    "enwik8_moe_unshared_xl_nonshared_schedule_lowreg_exp_drop_nopplreg",
    "enwik8_moe_unshared_xl_nonshared_schedule_lowreg_exp_drop_2x_64",
    "enwik8_moe_unshared_xl_nonshared_schedule_lowreg_exp_drop_k8_64",
    #----
    "wikitext103_small_moe_unshared_xl_nonshared_schedule_short_128_lowreg_k2_256",
    "enwik8_moe_unshared_xl_nonshared_schedule_lowreg_exp_drop_k2_256",

    "wikitext103_moe_smallbig",
    "wikitext103_moe_smallbig_softmax_after",
    "wikitext103_moe_smallbig_bad_init",
    "wikitext103_moe_smallbig_noreg",
    "wikitext103_moe_smallbig_k",
    "wikitext103_moe_smallbig_k1_512",
    "wikitext103_moe_smallbig_sinkmoid",
    "wikitext103_moe_smallbig_sinkmoid_h1",
    #---
    "wikitext103_moe_smallbig_2x_64",
    "wikitext103_moe_smallbig_switch_4head",
    "wikitext103_moe_smallbig_dropout",
    "wikitext103_moe_smallbig_k2_256",
    "wikitext103_moe_smallbig_switch_small",
    "wikitext103_moe_smallbig_switch",

    "wikitext103_moe_big_drop_expert_dropout_local_test",
    "wikitext103_moe_smallbig_k8_64",
    #----
    "wikitext103_moe_big_softmax",
    "wikitext103_moe_big_switch",
    "wikitext103_moe_big_softmax_after_top_missing",
    "wikitext103_moe_big_switch_missing",
    "wikitext103_moe_big_standard_dropout",
    "wikitext103_moe_big_sinkmoid",
    "wikitext103_moe_big_sinkmoid_h1_512",
    "wikitext103_moe_big_drop_expert_dropout_bad_init",
    "wikitext103_moe_big_k2_es256",
    "wikitext103_moe_big_k1_es512",
    "wikitext103_moe_big_noreg",
    "wikitext103_moe_big_k8_es64"
])


config_filters = [
("$\\sigma$-MoE (ours)", {
    "transformer.variant": "preln_moe",
    "pkm.n_heads": 4,
    "moe.selection_mode": "sigmoid",
    "moe.expert_size_init": False,
    "moe.expert_size": 128,
    "moe.reg_type": "entropy",
    "moe.perplexity_reg": lambda x, _: x>0,
    "kvmem.dropout": "none"
}),

("\\quad standard dropout", {
    "transformer.variant": "preln_moe",
    "pkm.n_heads": 4,
    "moe.selection_mode": "sigmoid",
    "moe.expert_size_init": False,
    "moe.expert_size": 128,
    "moe.reg_type": "entropy",
    "moe.perplexity_reg":  lambda x, _: x>0,
    "kvmem.dropout": "score"
}),


("\\quad softmax (after top-k)",  {
    "transformer.variant": "preln_moe",
    "pkm.n_heads": 4,
    "moe.selection_mode": "gate",
    "moe.expert_size_init": False,
    "moe.activation_after_topk": True
}),

("\\quad softmax (before top-k)", {
    "transformer.variant": "preln_moe",
    "pkm.n_heads": 4,
    "moe.selection_mode": "gate",
    "moe.expert_size_init": False,
    "moe.activation_after_topk": False,
    "moe.reg_type": "entropy",
}),

("\\quad standard init", {
    "transformer.variant": "preln_moe",
    "pkm.n_heads": 4,
    "moe.selection_mode": "sigmoid",
    "moe.expert_size_init": True,
}),

("\\quad no reg ($\\gamma=0,\\delta=0$)", {
    "transformer.variant": "preln_moe",
    "pkm.n_heads": 4,
    "moe.selection_mode": "sigmoid",
    "moe.expert_size_init": False,
    "moe.expert_size": 128,
    "moe.reg_type": "entropy",
    "moe.perplexity_reg": 0.0,
    "moe.drop_expert": 0
}),


("\\quad $K=8, G=64$", {
    "transformer.variant": "preln_moe",
    "pkm.n_heads": 8,
    "moe.selection_mode": "sigmoid",
    "moe.expert_size_init": False,
    "moe.expert_size": 64,
}),


("\\quad $K=2, G=256$", {
    "transformer.variant": "preln_moe",
    "pkm.n_heads": 2,
    "moe.selection_mode": "sigmoid",
    "moe.expert_size_init": False,
    "moe.expert_size": 256,
}),


("\\quad $K=1, G=512$", {
    "transformer.variant": "preln_moe",
    "pkm.n_heads": 1,
    "moe.selection_mode": "sigmoid",
    "moe.expert_size_init": False,
    "moe.expert_size": 512,
}),

("\\quad $N_E' = 2N_E, G=64$", {
    "transformer.variant": "preln_moe",
    "pkm.n_heads": 4,
    "moe.selection_mode": "sigmoid",
    "moe.expert_size_init": False,
    "moe.expert_size": 64
}),

("\\quad $K=1$", {
    "transformer.variant": "preln_moe",
    "pkm.n_heads": 1,
    "moe.selection_mode": "sigmoid",
    "moe.expert_size_init": False,
    "moe.expert_size": 128,
}),



("\\quad $K=2$", {
    "transformer.variant": "preln_moe",
    "pkm.n_heads": 2,
    "moe.selection_mode": "sigmoid",
    "moe.expert_size_init": False,
    "moe.expert_size": 128,
}),

("\\quad $K=8$", {
    "transformer.variant": "preln_moe",
    "pkm.n_heads": 8,
    "moe.selection_mode": "sigmoid",
    "moe.expert_size_init": False,
    "moe.expert_size": 128,
}),



(None, None),

("Switch, $K=1, G=512$", {
    "transformer.variant": "preln_moe",
    "pkm.n_heads": 1,
    "moe.selection_mode": "gate",
    "moe.expert_size": 512,
    "moe.reg_type": "switch",
    "moe.activation_after_topk": False,
    "moe.perplexity_reg": 0.01,
    "kvmem.dropout": "score"
}),

("\\quad no dropout", {
    "transformer.variant": "preln_moe",
    "pkm.n_heads": 1,
    "moe.selection_mode": "gate",
    "moe.expert_size": 512,
    "moe.reg_type": "switch",
    "moe.activation_after_topk": False,
    "moe.perplexity_reg": 0.01,
    "kvmem.dropout": "none"
}),

("\\quad $K=4, G=128$", {
    "transformer.variant": "preln_moe",
    "pkm.n_heads": 4,
    "moe.selection_mode": "gate",
    "moe.expert_size": 128,
    "moe.reg_type": "switch",
    "moe.activation_after_topk": False,
    "moe.perplexity_reg": 0.01
}),

("\\quad $K=1, G=128$", {
    "transformer.variant": "preln_moe",
    "pkm.n_heads": 1,
    "moe.selection_mode": "gate",
    "moe.expert_size": 128,
    "moe.reg_type": "switch",
    "moe.activation_after_topk": False,
    "moe.perplexity_reg": 0.01,
    "kvmem.dropout": "score"
}),

("\\quad\\quad no dropout", {
    "transformer.variant": "preln_moe",
    "pkm.n_heads": 1,
    "moe.selection_mode": "gate",
    "moe.expert_size": 128,
    "moe.reg_type": "switch",
    "moe.activation_after_topk": False,
    "moe.perplexity_reg": 0.01,
    "kvmem.dropout": "none"
}),

(None, None),

("S-BASE", {
    "transformer.variant": "preln_moe",
    "pkm.n_heads": 4,
    "moe.selection_mode": "sinkmoid2",
    "moe.expert_size_init": False,
    "moe.expert_size": 128,
    "moe.reg_type": "entropy",
    "moe.perplexity_reg": lambda x, _: x>0,
    "kvmem.dropout": "none"
}),

("\\quad $K=1, G=512$", {
    "transformer.variant": "preln_moe",
    "pkm.n_heads": 1,
    "moe.selection_mode": "sinkmoid2",
    "moe.expert_size_init": False,
    "moe.reg_type": "entropy",
    "moe.perplexity_reg": lambda x, _: x>0,
    "kvmem.dropout": "none"
}),


# []



# ("$\\gamma=0$", {
#     "transformer.variant": "preln_moe",
#     "pkm.n_heads": 4,
#     "moe.selection_mode": "sigmoid",
#     "moe.expert_size_init": False,
#     "moe.expert_size": 128,
#     "moe.reg_type": "entropy",
#     "moe.perplexity_reg": 0.0,
#     "moe.drop_expert": 0.05
# }),


]




# config_filters["bad init"] = {
#     "transformer.variant": "preln_moe",
#     "pkm.n_heads": 4,
#     "moe.selection_mode": "sigmoid",
#     "moe.expert_size_init": True
# }


dataset_filters = [
("Wikitext 103",  {
    "task": "wikitext103_sp_transformer",
    "state_size": 412,
    "moe.n_experts": lambda _, config: config["moe.n_experts"] * config["moe.expert_size"] <= 2048
}),

("Wikitext 103", {
    "task": "wikitext103_sp_transformer",
    "state_size": 412,
    "moe.n_experts": lambda _, config: config["moe.n_experts"] * config["moe.expert_size"] > 2048
}),

("Wikitext 103", {
    "task": "wikitext103_sp_transformer",
    "state_size": 1024,
}),

# dataset_filters["Wikitext 103 big"] = {
#     "task": "wikitext103_sp_transformer",
#     "state_size": 1024
# }

("Enwik8", {
    "task": "enwik8_transformer",
    "state_size": 512,
}),
]


# baselines = {

# }


defaults = {
    "moe.expert_size_init": False
}


def find_config(runs, filter):
    found = None
    for r in runs:
        for k, v in filter.items():
            if callable(v):
                if not v(r.config.get(k, defaults.get(k)), r.config):
                    break
            else:
                if r.config.get(k, defaults.get(k)) != v:
                    # print("Mismatch", k, r.config[k], v)
                    break
        else:
            if found is None:
                found = r
            else:
                raise Exception("Multiple runs found")
    return found.id if found is not None else None

tasks = []

first_model_sizes = []
for dataset_name, dfilter in dataset_filters:
    f = {**dfilter, **config_filters[0][1]}
    id = find_config(runs, f)
    first_model_sizes.append(int(round(infos[id]["n_param"]/1e6)))


print("\\begin{tabular}{lrr" + "r" * len(dataset_filters) + "}")
print("\\toprule")
print("Dataset & & & " + " & ".join(df[0] for df in dataset_filters) + " \\\\")
print("$\\dmodel$ & & &" + " & ".join(str(df[1]["state_size"]) for df in dataset_filters) + " \\\\")
print("\\# params & & &" + " & ".join(f"{s}M" for s in first_model_sizes) + " \\\\")
print("& G & K " + " & "*len(first_model_sizes) + " \\\\")
print("\\midrule")

for config_name, cfilter in config_filters:
    if cfilter is None:
        print("\\midrule")
        continue
    row  = []
    k = None
    g = None
    for dataset_name, dfilter in dataset_filters:
        id = find_config(runs, {**cfilter, **dfilter})
        if id is not None:
            # res = f"{infos[id]['test_results'][test_field_name[dfilter['task']]]:.2f}"
            # res = runs[id].config["pkm.n_heads"] / runs[id].config["moe.n_experts"]
            thisrun = [r for r in runs if r.id == id][0]
            if k is None:
                k = thisrun.config["pkm.n_heads"]
                g = thisrun.config["moe.expert_size"]
            else:
                assert k == thisrun.config["pkm.n_heads"]
                assert g == thisrun.config["moe.expert_size"]

            n = thisrun.config["moe.n_experts"]
        else:
            res = "-"

        row.append(f"{k/n*100:.1f}\\%")

    print(f"{config_name} & {g} & {k} & {' & '.join(row)} \\\\")
print("\\bottomrule")
print("\\end{tabular}")


def print_sep(sep="", sep2=""):
    print(f"| {sep} | {sep} | {sep} | {sep2} " + (f" | {sep2} ").join(" " for df in dataset_filters) + " |")

print("| Dataset | | | " + " | ".join(df[0] for df in dataset_filters) + " |")
print_sep("---", "---:")
print("| $d_{model}$ | | | " + " | ".join(str(df[1]["state_size"]) for df in dataset_filters) + " |")
print("| # params | | | " + " | ".join(f"{s}M" for s in first_model_sizes) + " |")
print("| | G | K " + " | "*len(first_model_sizes) + " |")
print_sep()

for config_name, cfilter in config_filters:
    if cfilter is None:
        print_sep()
        continue
    row  = []
    k = None
    g = None
    for dataset_name, dfilter in dataset_filters:
        id = find_config(runs, {**cfilter, **dfilter})
        if id is not None:
            # res = f"{infos[id]['test_results'][test_field_name[dfilter['task']]]:.2f}"
            # res = runs[id].config["pkm.n_heads"] / runs[id].config["moe.n_experts"]
            thisrun = [r for r in runs if r.id == id][0]
            if k is None:
                k = thisrun.config["pkm.n_heads"]
                g = thisrun.config["moe.expert_size"]
            else:
                assert k == thisrun.config["pkm.n_heads"]
                assert g == thisrun.config["moe.expert_size"]

            n = thisrun.config["moe.n_experts"]
        else:
            res = "-"

        row.append(f"{k/n*100:.1f}\\%")

    n = config_name.replace("\\quad", "$~~~$")
    print(f"| {n} | {g} | {k} | {' | '.join(row)} |")
print_sep()
