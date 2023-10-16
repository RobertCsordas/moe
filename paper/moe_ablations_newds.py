from collections import OrderedDict
from run_tests import get_runs_and_infos, dataset, n_test_blocks, test_field_name


 #"enwik8_moe_unshared_xl_nonshared_schedule_lowreg_exp_drop_bad_init"
runs, infos = get_runs_and_infos([
    "c4_xl",
    "c4_moe",
    "c4_moe_switch",
    "c4_sbase",

    "pes2o_xl",
    "pes2o_moe",
    "pes2o_moe_switch",
    "pes2o_sbase",

    "c4_baseline_big",
    "c4_moe_big",
    "c4_big_switch",
    "c4_big_sinkmoid",

    "pes2o_baseline_big",
    "pes2o_moe_big",
    "pes2o_big_switch",
    "pes2o_big_sinkmoid"
])

# config_filters = OrderedDict()
config_filters = [
("Dense", {
    "transformer.variant": "preln_relative",
}),

("$\\sigma$-MoE (ours)", {
    "transformer.variant": "preln_moe",
    "pkm.n_heads": 4,
    "moe.selection_mode": "sigmoid",
    "moe.expert_size_init": False,
    "moe.expert_size": 128,
    "moe.reg_type": "entropy",
    # "moe.perplexity_reg": lambda x, _: x>0,
    "kvmem.dropout": "none"
}),


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



]




# config_filters["bad init"] = {
#     "transformer.variant": "preln_moe",
#     "pkm.n_heads": 4,
#     "moe.selection_mode": "sigmoid",
#     "moe.expert_size_init": True
# }


dataset_filters = [
("C4",  {
    "task": "c4_transformer",
    "state_size": 412,
}),


("C4", {
    "task": "c4_transformer",
    "state_size": 1024,
}),

("peS2o", {
    "task": "pes2o_transformer",
    "state_size": 412,
}),


("peS2o", {
    "task": "pes2o_transformer",
    "state_size": 1024,
}),

]


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
    if id is None:
        first_model_sizes.append("-")
    else:
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
            res = f"{infos[id]['test_results'][test_field_name[dfilter['task']]]:.2f}"
            thisrun = [r for r in runs if r.id == id][0]
            if k is None:
                k = thisrun.config["pkm.n_heads"]
                g = thisrun.config["moe.expert_size"]
            else:
                assert k == thisrun.config["pkm.n_heads"]
                assert g == thisrun.config["moe.expert_size"]
        else:
            res = "-"

        row.append(res)

    print(f"{config_name} & {g} & {k} & {' & '.join(row)} \\\\")
print("\\bottomrule")
print("\\end{tabular}")



def print_sep(sep="", sep2=""):
    print(f"| {sep} | {sep} | {sep} | {sep2} " + (f" | {sep2} ").join(" " for df in dataset_filters) + " |")

print("|Dataset | | | " + " | ".join(df[0] for df in dataset_filters) + " |")
print_sep("---", "---:")
print("| $d_{model}$ | | | " + " | ".join(str(df[1]["state_size"]) for df in dataset_filters) + " |")
print("| # params | | | " + " |  ".join(f"{s}M" for s in first_model_sizes) + " |")
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
            res = f"{infos[id]['test_results'][test_field_name[dfilter['task']]]:.2f}"
            thisrun = [r for r in runs if r.id == id][0]
            if k is None:
                k = thisrun.config["pkm.n_heads"]
                g = thisrun.config["moe.expert_size"]
            else:
                assert k == thisrun.config["pkm.n_heads"]
                assert g == thisrun.config["moe.expert_size"]
        else:
            res = "-"

        row.append(res)

    n = config_name.replace("\\quad", "$~~~$")
    print(f"| {n} | {g} | {k} | {' | '.join(row)} |")
