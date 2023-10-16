import lib
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

coo = torch.load("coocurence.pt").cpu()

coo.fill_diagonal_(0)
coo = coo / coo.sum(-1, keepdim=True)

# n_experts = 4
# coo = coo / coo.diag().clamp(min=1).unsqueeze(-1)
# coo.fill_diagonal_(0)
# coo = coo / (n_experts - 1)
# sss = coo.sum(-1)

fig = plt.figure(figsize=(3, 3))
ax = plt.gca()

im = plt.imshow(coo, cmap=plt.cm.Blues)
x = [1, 5, 9, 13]
ax.set_xticks([xx-1 for xx in x], x)
ax.set_yticks([xx-1 for xx in x], x)
plt.ylabel("Expert (target)")
plt.xlabel("Expert (used with)")

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)

plt.savefig("coocurence.pdf", bbox_inches="tight")