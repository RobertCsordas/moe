import torch
import time
from moe_layer_simple import MoE
from dataclasses import dataclass
import lib


bs = 64
len = 512

expert_size_d = 128
n_experts_d = 32
n_heads_d = 4
n_channels_d = 512

n_test = 10


device = torch.device('cuda:0')


@dataclass
class BenchmarResult:
    duration_ms: float
    memory_mb: float


def benchmark(expert_size: int, n_experts: int, n_heads: int, n_channels: int):
    input = torch.randn(bs, len, n_channels, device=device, requires_grad=True)
    gin = torch.randn(bs, len, n_channels, device=device)
    gloss = torch.tensor([1], device=device)


    class MLP(torch.nn.Module):
        def __init__(self, dmodel: int, dff: int) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(dmodel, dff, bias=False),
                torch.nn.ReLU(),
                torch.nn.Linear(dff, dmodel, bias=False)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)



    mlp = MLP(n_channels, n_experts * expert_size).to(device)
    moe = MoE(n_channels, n_experts, expert_size, n_heads).to(device)

    mlp(input).backward(gin, retain_graph=True)
    moe(input)[0].backward((gin, gloss), retain_graph=True)

    # print(input.grad)


    torch.cuda.synchronize()
    start = time.time()

    for i in range(n_test):
        mlp(input).backward(gin, retain_graph=True)
        torch.cuda.synchronize()

    end = time.time()

    len_mlp = (end - start) / n_test * 1000

    torch.cuda.synchronize()
    start = time.time()

    for i in range(n_test):
        moe(input)[0].backward((gin, gloss), retain_graph=True)
        torch.cuda.synchronize()

    end = time.time()

    len_moe = (end - start) / n_test * 1000

    torch.cuda.empty_cache()

    used_mem_mlp = torch.cuda.memory_allocated()
    x = mlp(input)
    used_mem_mlp = (torch.cuda.memory_allocated() - used_mem_mlp) / 1024 / 1024
    x.backward(gin)
    del x

    torch.cuda.empty_cache()
    used_mem_moe = torch.cuda.memory_allocated()
    x = moe(input)[0]
    used_mem_moe = (torch.cuda.memory_allocated() - used_mem_moe) / 1024 / 1024
    x.backward((gin, gloss))
    del x

    return BenchmarResult(len_mlp, used_mem_mlp), BenchmarResult(len_moe, used_mem_moe)



import matplotlib.pyplot as plt
import numpy as np


def plot(fname, x, res, xlabel):
    fig, ax = plt.subplots(figsize=(3, 1))
    ax2 = ax.twinx()
    ax.plot(x, [r[0].duration_ms for r in res], label="MLP", marker=".")
    ax.plot(x, [r[1].duration_ms for r in res], label="MoE", marker=".")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Time (ms)")
    ax.legend(loc="upper left")
    plt.xticks(x, [str(y) for y in x], rotation=90)

    # fig.savefig("benchmark_time.pdf", bbox_inches='tight')

    # fig, ax = plt.subplots()
    ax2.plot(x, [r[0].memory_mb/1024 for r in res], "--", label="MLP")
    ax2.plot(x, [r[1].memory_mb/1024 for r in res], "--", label="MoE")
    ax2.set_ylabel("Memory (GB)")
    ax2.legend(loc="upper left")

    fig.savefig(fname, bbox_inches='tight', pad_inches = 0.01)
    #  fig.close()

n_experts = [8, 16, 32, 64, 128, 256]
res = [benchmark(expert_size_d, ne, n_heads_d, n_channels_d) for ne in n_experts]


plot("benchmark.pdf", n_experts, res, "Number of experts ($N_E$)")



# # expert_size = [64, 128, 256, 512, 1024]
# # res = [benchmark(es, n_experts_d, n_heads_d, n_channels_d) for es in expert_size]

# # plot("benchmark_dmodel.pdf", n_experts, res, "Expert size ($G$)")

# expert_size = [64, 128, 256, 512, 1024]
# res = [benchmark(es, n_experts_d, n_heads_d, n_channels_d) for es in expert_size]

# plot("benchmark_es.pdf", expert_size, res, "Expert size ($G$)")


# n_channels = [128, 256, 512, 1024, 2048]
# res = [benchmark(expert_size_d, n_experts_d, n_heads_d, nc) for nc in n_channels]

# plot("benchmark_dmodel.pdf", n_channels, res, "$d_\\mathrm{model}$")

# import matplotlib.pyplot as plt
# import numpy as np

# fig, ax = plt.subplots()
# ax.plot(expert_size, [r[0].duration_ms for r in res], label="MLP")
# ax.plot(expert_size, [r[1].duration_ms for r in res], label="MoE")
# ax.set_xlabel("Expert size")
# ax.set_ylabel("Time (ms)")
# ax.set_title("Time")
# ax.legend()

# fig.savefig("benchmark_time_es.pdf", bbox_inches='tight')

# fig, ax = plt.subplots()
# ax.plot(expert_size, [r[0].memory_mb for r in res], label="MLP")
# ax.plot(expert_size, [r[1].memory_mb for r in res], label="MoE")
# ax.set_xlabel("Expert size")
# ax.set_ylabel("Memory (MB)")
# ax.set_title("Memory")
# ax.legend()

# fig.savefig("benchmark_memory_es.pdf", bbox_inches='tight')
