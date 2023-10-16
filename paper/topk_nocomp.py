from run_tests import get_runs_and_infos, dataset, n_test_blocks, test_field_name


runs, infos = get_runs_and_infos([
    "wikitext103_baseline_big",
    "wikitext103_baseline_big_topk",

    "enwik8_baseline",
    "enwik8_baseline_topk_nocomp",

    "wikitext103_small_xl_nonshared_schedule_short_matched_topk_nocomp",
    "wikitext103_small_xl_nonshared_schedule_short_matched"])


sort_keys=["task", "n_params", "transformer.variant", "transformer.topk_value"]
order = list(sorted(range(len(runs)), key=lambda i: tuple(infos[runs[i].id]["config"][k] for k in sort_keys)))

prev = None
print("\\begin{tabular}{lrrrr}")
print("\\toprule")
print("Dataset & \\#params & $d_{ff}$ & K & bpc/perplexity \\\\")
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

    print(f"{dataset[run.config['task']]} & {n_params}M & {dff} & {k} & {infos[run.id]['test_results'][test_field_name[run.config['task']]]:.2f} \\\\")

# wikitext103_small_xl_nonshared_schedule_short_matched_topk


        #  "enwik8_baseline", "enwik8_baseline_topk"]

print("\\bottomrule")
print("\\end{tabular}")
