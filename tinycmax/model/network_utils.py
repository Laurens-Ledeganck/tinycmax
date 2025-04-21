from collections.abc import Sequence

from dotmap import DotMap
import torch
import torch.nn as nn


def recursive_detach(x):
    """
    Detach all tensors in a nested structure.
    """
    if isinstance(x, torch.Tensor):
        x.detach_()
    elif isinstance(x, Sequence):
        [recursive_detach(xx) for xx in x]
    elif x is None:
        pass
    else:
        raise TypeError(f"expected torch.Tensor or Sequence, got {type(x)}")


def recursive_clone(x):
    """
    Clone all tensors in a nested structure.
    """
    if isinstance(x, torch.Tensor):
        return x.clone()
    elif isinstance(x, Sequence):
        return [recursive_clone(xx) for xx in x]
    elif isinstance(x, DotMap):
        return DotMap({k: recursive_clone(v) for k, v in x.items()})
    elif x is None:
        return None
    else:
        raise TypeError(f"expected torch.Tensor or Sequence, got {type(x)}")


class NetworkWrapper(nn.Module):
    """
    Wrapper for ease of use during training, while allowing the base
    network to be compiled with e.g. TensorRT.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.reset_state()  # TODO: register as buffer? tensordict?

    def forward(self, input):
        hidden = self.get_state()
        x, hidden = super().forward(input, hidden)
        self.set_state(hidden)
        return x

    def trace(self, input, device="cpu"):
        input = DotMap({k: v.to(device) for k, v in input.items()}, _dynamic=False)
        with torch.no_grad():
            self(input)
        self.reset()

    def reset(self):
        self.reset_state()

    def detach(self):
        self.detach_state()

    def reset_state(self):
        self.state = None

    def detach_state(self):
        recursive_detach(self.state)

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state  # TODO: why does clone here not work?
