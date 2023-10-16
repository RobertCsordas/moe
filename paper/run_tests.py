import os
import lib
import json
import copy
from typing import List
import wandb
import shutil

my_dir = os.path.dirname(__file__)
main_dir = os.path.abspath(my_dir+"/../..")
my_rel_dir = os.path.relpath(my_dir, main_dir)
curr_dir = os.getcwd()


def get_info(id, test_blocks: int):
    dest_dir = f"checkpoints/{id}/"
    res_path = f"{dest_dir}/result.json"

    config = lib.get_config()

    if not os.path.isfile(res_path):
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
        cmd = f"python3 main.py --name post_validate --log tb --restore {my_rel_dir}/{dest_dir}/model-100000.pth --test_only 1  -reset 1 -lm.trafo.test_context_blocks {test_blocks} --keep_alive 0 --test_batch_size 4"
        print("Validate command: ", cmd)
        out = lib.run_command(cmd)
        lines = out.splitlines()
        start_line = lines.index('Validate returned:')
        end_line = None
        for i in range(start_line, len(lines)):
            if lines[i].startswith("-------"):
                end_line = i
                break

        assert end_line is not None

        n_param = None
        for l in lines:
            if l.startswith("Total number of model parameters"):
                n_param = int(l.split(":")[1])

        assert n_param is not None

        res = "\n".join(lines[start_line+1:end_line])
        os.chdir(curr_dir)

        with open(res_path, "w") as f:
            f.write(json.dumps({
                "n_param": n_param,
                "test_results": json.loads(res)
            }))

    with open(res_path, "r") as f:
        res = json.load(f)

    return res


dataset = {
    "wikitext103_sp_transformer": "WikiText 103",
    "enwik8_transformer": "Enwik8"
}

n_test_blocks = {
    "wikitext103_sp_transformer": 4,
    "enwik8_transformer": 4
}

test_field_name = {
    "wikitext103_sp_transformer": "test/perplexity",
    "enwik8_transformer": "test/bpc"
}


def get_runs_and_infos(sweep_names: List[str]):
    runs = lib.get_runs(sweep_names, check_finished=False)

    print("Running tests...")

    infos = {r.id: get_info(r.id, n_test_blocks[r.config["task"]]) for r in runs}

    for r in runs:
        c = copy.deepcopy(r.config)
        c["n_params"] = infos[r.id]["n_param"]
        c["n_params_m"] = round(infos[r.id]["n_param"] / 1000000)
        infos[r.id]["config"] = c

    print("Done.")

    return runs, infos

