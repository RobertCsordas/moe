from run_tests import get_runs_and_infos, dataset, n_test_blocks, test_field_name


runs, infos = get_runs_and_infos(["wikitext103_baseline_big", "wikitext103_moe_big_drop_expert", "enwik8_moe_unshared_xl_nonshared_schedule_lowreg_exp_drop", "enwik8_baseline", "wikitext103_small_moe_unshared_xl_nonshared_schedule_short_128", "wikitext103_small_xl_nonshared_schedule_short_matched"])


sort_keys=["task", "n_params", "transformer.variant"]
order = list(sorted(range(len(runs)), key=lambda i: tuple(infos[runs[i].id]["config"][k] for k in sort_keys)))

model = {
    "preln_relative": "Dense",
    "preln_moe": "MoE",
}

prev = None
print("\\begin{tabular}{llrrr}")
print("\\toprule")
print("Dataset & Model & \\#params & \\% FF params used at once & bpc/perplexity \\\\")
print("\\midrule")
for o in order:
    run = runs[o]

    k = run.config["transformer.topk_value"] if "topk" in run.config["transformer.variant"] else "-"
    n_params = infos[run.id]["config"]["n_params_m"]

    dff = int(run.config["state_size"] * run.config["transformer.ff_multiplier"])

    id = (run.config['task'], n_params)
    if prev is not None and prev != id:
        print("\\midrule")
    prev = id

    if "moe" in run.config["transformer.variant"]:
        pct_used = round(infos[run.id]["config"]["pkm.n_heads"] * 100 / infos[run.id]["config"]["moe.n_experts"], 1)
    else:
        pct_used = 100

    print(f"{dataset[run.config['task']]} & {model[run.config['transformer.variant']]} & {n_params}M & {pct_used}\\% & {infos[run.id]['test_results'][test_field_name[run.config['task']]]:.2f} \\\\")

# wikitext103_small_xl_nonshared_schedule_short_matched_topk


        #  "enwik8_baseline", "enwik8_baseline_topk"]

print("\\bottomrule")
print("\\end{tabular}")
