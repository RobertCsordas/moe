from collections import OrderedDict
from run_tests import get_runs_and_infos, dataset, n_test_blocks, test_field_name


 #"enwik8_moe_unshared_xl_nonshared_schedule_lowreg_exp_drop_bad_init"
runs, infos = get_runs_and_infos([
    "enwik8_baseline",
    "wikitext103_small_xl_nonshared_schedule_short_matched",
    "enwik8_pkm",
    "wikitext103_pkm_softmax",
    "wikitext103_pkm",
    "wikitext103_pkm_param_matched",
    "wikitext103_baseline_big",
    "wikitext103_big_pkm",
    "wikitext103_big_pkm_match_missing",
    "wikitext103_big_pkm_match",
    "enwik8_pkm_matched",
    "wikitext103_big_pkm_match_init",
    "enwik8_pkm_matched_init",
    "wikitext103_pkm_param_matched_init"
])

config_filters = OrderedDict()

config_filters["Baseline"] = {
    "transformer.variant": "preln_relative",
}


config_filters["PKM, value-count matched"] = {
    "transformer.variant": "preln_kvmem",
    "kvmem.approx_topk": False,
    "transformer.activation": "softmax",
    "pkm.n_keys": lambda x: x in {46, 64}
}

config_filters["PKM, param matched"] = {
    "transformer.variant": "preln_kvmem",
    "kvmem.approx_topk": False,
    "transformer.activation": "relu",
    "pkm.n_keys": lambda x: x in {62, 89},
    "pkm.custom_init": 0
}

config_filters["PKM, param matched + init"] = {
    "transformer.variant": "preln_kvmem",
    "kvmem.approx_topk": False,
    "transformer.activation": "relu",
    "pkm.n_keys": lambda x: x in {62, 89},
    "pkm.custom_init": 5
}

print(infos)


dataset_filters = OrderedDict()
dataset_filters["Wikitext 103 small"] = {
    "task": "wikitext103_sp_transformer",
    "state_size": 412
}

dataset_filters["Wikitext 103 big"] = {
    "task": "wikitext103_sp_transformer",
    "state_size": 1024
}

dataset_filters["Enwik8"] = {
    "task": "enwik8_transformer",
}


defaults = {
    "moe.expert_size_init": False
}


def find_config(runs, filter):
    found = None
    for r in runs:
        for k, v in filter.items():
            if callable(v):
                if not v(r.config.get(k, defaults.get(k))):
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


print("\\begin{tabular}{ll" + "r" * len(dataset_filters) + "}")
print("\\toprule")
print("Variant & Nonlinearity & " + " & ".join(dataset_filters.keys()) + " \\\\")
print("\\midrule")

for config_name, cfilter in config_filters.items():
    for nonlinearity in ["Softmax", "ReLU"]:
        row  = []
        cf2 = cfilter.copy()
        cf2["transformer.activation"] = nonlinearity.lower()
        row.append(nonlinearity)
        if config_name == "Baseline" and nonlinearity != "ReLU":
            continue

        for dataset_name, dfilter in dataset_filters.items():
            id = find_config(runs, {**cf2, **dfilter})
            if id is not None:
                res = f"{infos[id]['test_results'][test_field_name[dfilter['task']]]:.2f}"
            else:
                res = "-"

            row.append(res)

        print(f"{config_name} & {' & '.join(row)} \\\\")
print("\\bottomrule")
print("\\end{tabular}")

# sort_keys=["task", "n_params", "transformer.variant"]
# order = list(sorted(range(len(runs)), key=lambda i: tuple(infos[runs[i].id]["config"][k] for k in sort_keys)))

# model = {
#     "preln_relative": "Dense",
#     "preln_moe": "MoE",
# }

# prev = None
# print("\\begin{tabular}{llrrr}")
# print("\\toprule")
# print("Dataset & Model & \\#params & \\% FF params used at once & bpc/perplexity \\\\")
# print("\\midrule")
# for o in order:
#     run = runs[o]

#     k = run.config["transformer.topk_value"] if "topk" in run.config["transformer.variant"] else "-"
#     n_params = infos[run.id]["config"]["n_params_m"]

#     dff = int(run.config["state_size"] * run.config["transformer.ff_multiplier"])

#     id = (run.config['task'], n_params)
#     if prev is not None and prev != id:
#         print("\\midrule")
#     prev = id

#     if "moe" in run.config["transformer.variant"]:
#         pct_used = round(infos[run.id]["config"]["pkm.n_heads"] * 100 / infos[run.id]["config"]["moe.n_experts"], 1)
#     else:
#         pct_used = 100

#     print(f"{dataset[run.config['task']]} & {model[run.config['transformer.variant']]} & {n_params}M & {pct_used}\\% & {infos[run.id]['test_results'][test_field_name[run.config['task']]]:.2f} \\\\")

# # wikitext103_small_xl_nonshared_schedule_short_matched_topk


#         #  "enwik8_baseline", "enwik8_baseline_topk"]

# print("\\bottomrule")
# print("\\end{tabular}")
