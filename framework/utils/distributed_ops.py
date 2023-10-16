import torch
import torch.distributed


def logsumexp(x: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    if not torch.distributed.is_initialized():
        return x.logsumexp(dim=dim, keepdim=keepdim)

    # Calculate numerically stable distributed logsumexp
    xmax = x.max(dim=dim, keepdim=True).values
    torch.distributed.all_reduce(xmax, op=torch.distributed.ReduceOp.MAX)

    xe = (x - xmax).exp().sum(dim=dim, keepdim=True)
    torch.distributed.all_reduce(xe, op=torch.distributed.ReduceOp.SUM)

    res = (xmax + xe.log())
    if not keepdim:
        res = res.squeeze(dim)

    return res
