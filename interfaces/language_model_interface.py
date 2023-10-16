import torch
import torch.nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from .model_interface import ModelInterface
from .result import RecurrentResult
from framework.utils import U
import random
from framework.helpers.distributed import DistributedEnv


class LanguageModelResult(RecurrentResult):
    @property
    def batch_size(self) -> int:
        o = self.outputs.output if isinstance(self.outputs, torch.nn.modules.adaptive._ASMoutput) else self.outputs
        return o.shape[self.batch_dim]


class LanguageModelInterface(ModelInterface):
    def __init__(self, model: torch.nn.Module, batch_dim: int = 1, drop_state_prob: float = 0,
                 dist_env: Optional[DistributedEnv] = None, save_state: bool = False):
        super().__init__()
        self.model = model
        self.state = None
        self.batch_dim = batch_dim
        self.drop_state_prob = drop_state_prob
        self.time_dim = 1 - self.batch_dim
        self.dist_env = dist_env
        self.save_state = save_state

    def create_input(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        return data["data"].narrow(self.time_dim, 0, data["data"].shape[self.time_dim] - 1)

    def decode_outputs(self, outputs: RecurrentResult) -> Any:
        return outputs.outputs

    def reset_state(self):
        self.state = None

    def loss(self, net_out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert net_out.shape[:-1] == target.shape
        return F.cross_entropy(net_out.flatten(0, -2), target.flatten().long())

    def create_target(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        return data["data"].narrow(self.time_dim, 1, data["data"].shape[self.time_dim] - 1).contiguous()

    def __call__(self, data: Dict[str, torch.Tensor]) -> LanguageModelResult:
        if self.model.training and self.drop_state_prob > 0 and random.random() < self.drop_state_prob:
            self.state = None

        input = self.create_input(data)
        target = self.create_target(data)

        res, state = self.model(input, target, self.state)
        if isinstance(res, torch.nn.modules.adaptive._ASMoutput):
            loss = res.loss
            # res = res.outputs
        else:
            loss = self.loss(res, target)

        self.state = U.apply_to_tensors(state, lambda x: x.detach())
        return LanguageModelResult(res, loss)

    def state_dict(self) -> Dict[str, Any]:
        if not self.save_state:
            return {}

        if self.dist_env is not None and self.dist_env.is_distributed:
            # Collect the state from all workers
            alist = [None] * self.dist_env.world_size
            state = torch.distributed.all_gather(alist, self.state)
            state = torch.cat(state, self.batch_dim)
            return {"state": state}
        else:
            return {"state": self.state}

    def load_state_dict(self, state: Dict[str, Any]):
        if not self.save_state:
            self.state = None
            return

        if self.dist_env is not None and self.dist_env.is_distributed:
            state_bs = state["state"].shape[self.batch_dim]
            if state_bs % self.dist_env.world_size != 0:
                print(f"WARNING: State batch size ({state_bs}) is not divisible by the number of workers ({self.dist_env.world_size}). Resetting state.")
                self.state = None
            else:
                bs_per_worker = state_bs // self.dist_env.world_size
                self.state = state["state"].narrow(self.batch_dim, self.dist_env.local_rank * bs_per_worker, bs_per_worker)
        else:
            self.state = state["state"]
