import os
import typing
import torch

class CheckPoint:
    def save_checkpoint(self, model : torch.nn.Module, optimizer : torch.optim.Optimizer, iteration : int, out : str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
        dic = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "iteration": iteration
        }
        torch.save(dic, out)

    def load_checkpoint(self, src, model : torch.nn.Module, optimizer : torch.optim.Optimizer):
        dic = torch.load(src)
        model.load_state_dict(dic["model_state"])
        optimizer.load_state_dict(dic["optimizer_state"])
        return dic["iteration"]



