import lib
import matplotlib.pyplot as plt
import torch

counts = torch.load("counts.pth")
# counts = torch.load("counts_big.pth")
# counts = torch.load("counts_enwik8.pth")
means = counts["means"]
stds = counts["stds"]

print(means)

# plt.figure(figsize=(5, 2.5))
plt.figure(figsize=(5, 1.5))
plt.bar(means.keys(), means.values(), yerr=stds.values())
plt.xticks(rotation=90)
plt.tight_layout()
plt.xlabel("Layer")
plt.xticks(range(0, len(means)), range(1, len(means) + 1))
plt.ylabel("Active channels")
plt.savefig("counts.pdf", bbox_inches='tight', pad_inches = 0.01)