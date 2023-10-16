import lib
import matplotlib.pyplot as plt
import torch
import os
import shutil
import wandb

my_dir = os.path.dirname(__file__)
main_dir = os.path.abspath(my_dir+"/../..")
my_rel_dir = os.path.relpath(my_dir, main_dir)
curr_dir = os.getcwd()


models = {
    "$\sigma$-MoE": ("wikitext103_moe_smallbig", {"moe.selection_mode": "sigmoid"}),
    "$\sigma$-MoE - softmax (no renorm.)": ("wikitext103_moe_smallbig", {"moe.selection_mode": "gate"}),
    "$\sigma$-MoE - softmax (renorm.)": ("wikitext103_moe_smallbig_softmax_after", {"moe.selection_mode": "gate"}),
    "Switch Transformer": ("wikitext103_moe_smallbig_switch_small", {"kvmem.dropout": "score"}),
    "S-BASE (K=4, G=128)": ("wikitext103_moe_smallbig_sinkmoid", {})
}


type = "soft"


def do_test(id):
    dest_dir = f"checkpoints/{id}/"
    res_path = f"{dest_dir}/result.json"
    counts_path = f"{dest_dir}/counts_val.pth"
    model_path = f"{dest_dir}/model-100000.pth"

    print(f"Testing {id}")

    if not os.path.isfile(counts_path):
        print(f"Counts missing: {counts_path}")
        if not os.path.isfile(model_path):
            config = lib.get_config()
            api = wandb.Api()
            run = api.run(config["wandb_project"] + "/" + id)

            n_iters = 100000 # run.summary["iteration"]

            f = run.file(f"checkpoint/model-{n_iters}.pth")
            if f.size != 0:
                print("Downloading checkpoint...")
                f.download(dest_dir)
                shutil.move(f"{dest_dir}/checkpoint/model-{n_iters}.pth", f"{dest_dir}/model-{n_iters}.pth")
                os.rmdir(f"{dest_dir}/checkpoint")
            else:
                print("Searching for checkpoint on the server...")
                lib.get_ckpt(id, dest_dir)

        os.chdir(main_dir)
        cmd = f"python3 main.py --name post_validate --log tb --restore {my_rel_dir}/{model_path} --test_only 1  -reset 1 --keep_alive 0 --test_batch_size 1 -moe.dump_usage_counts_prefix {my_rel_dir}/{dest_dir}/counts"
        print("Validate command: ", cmd)
        out = lib.run_command(cmd)

    os.chdir(my_dir)
    return torch.load(counts_path)[type]

runs_for_sweeps = {}

for sweep, _ in models.values():
    runs_for_sweeps[sweep] = lib.get_runs([sweep], check_finished=False)

os.makedirs("count_plots", exist_ok=True)

all_counts = {}
for n, (sweep, filter) in models.items():
    runs = [r for r in runs_for_sweeps[sweep] if all(r.config.get(k)==v for k, v in filter.items())]
    assert len(runs) == 1
    nheads = runs[0].config["pkm.n_heads"]
    counts = do_test(runs[0].id)
    counts = {k: sum(v) for k, v in counts.items()}
    counts = {k: v/v.sum() for k, v in counts.items()}
    # counts = {k: v/nheads for k, v in counts.items()}
    all_counts[n] = counts

ncol = 2
nrows = len(all_counts["$\sigma$-MoE"])//ncol
fig, ax = plt.subplots(nrows, ncol, figsize=(20, 30))


def do_plot(layer: str, plotxlabel: bool, plotylabel: bool, plottitle: bool = True):
    colors = ['blue','orange','green','red','darkgray']
    for j, model in enumerate(all_counts.keys()):
        counts = all_counts[model]
        sorted_counts = counts[layer].sort(descending=True).values.cpu().numpy()
        l = len(sorted_counts)
        plt.plot([i for i in range(l)], sorted_counts, label=model, color=colors[j])
        # plt.bar([str(i) for i in range(l)], sorted_counts, label=model, width=1, alpha=0.5)

    plt.xticks([i for i in range(l)], [str(i) if i % 16==0 else "" for i in range(l)])
    plt.yscale("log")
    if plotxlabel:
        plt.xlabel("Expert")

    if plotylabel:
        plt.ylabel("Selection proportion")

    # if r == 0 and c == 0:
    plt.legend()
    plt.xlim(-0.5, l-0.5)
    if plottitle:
        plt.title("Layer "+layer.split(".")[1])


l = list(sorted(all_counts["$\sigma$-MoE"].keys()))

plt.figure(figsize=(5, 2.5))
do_plot(l[11], True, True, False)
plt.savefig(f"count_plots/{l[11]}_{type}.pdf", bbox_inches='tight')


for i, layer in enumerate(all_counts["$\sigma$-MoE"]):
    # plt.figure()
    r = i // 2
    c = i % 2
    plt.sca(ax[r, c])
    print(f"Plotting {layer}")
    # plt.figure()
    do_plot(layer, r==nrows-1, c==0)

    # plt.savefig(f"count_plots/{layer}.pdf", bbox_inches='tight')
    # plt.close()

fig.savefig(f"count_plots/all_{type}.pdf", bbox_inches='tight')
