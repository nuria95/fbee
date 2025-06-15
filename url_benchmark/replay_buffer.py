# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" 
Code from  https://github.com/facebookresearch/controllable_agent
"""

import typing as tp
import dataclasses

import numpy as np
import torch


EpisodeTuple = tp.Tuple[np.ndarray, ...]
Episode = tp.Dict[str, np.ndarray]
T = tp.TypeVar("T", np.ndarray, torch.Tensor)
B = tp.TypeVar("B", bound="EpisodeBatch")


@dataclasses.dataclass
class EpisodeBatch(tp.Generic[T]):
    """For later use
    A container for batchable replayed episodes
    """
    obs: T
    action: T
    reward: T
    next_obs: T
    discount: T
    meta: tp.Dict[str, T] = dataclasses.field(default_factory=dict)
    _physics: tp.Optional[T] = None
    goal: tp.Optional[T] = None
    next_goal: tp.Optional[T] = None
    future_obs: tp.Optional[T] = None
    future_goal: tp.Optional[T] = None

    def __post_init__(self) -> None:
        # some security to be removed later
        assert isinstance(self.reward, (np.ndarray, torch.Tensor))
        assert isinstance(self.discount, (np.ndarray, torch.Tensor))
        assert isinstance(self.meta, dict)

    def to(self, device: str) -> "EpisodeBatch[torch.Tensor]":
        """Creates a new instance on the appropriate device"""
        out: tp.Dict[str, tp.Any] = {}
        for field in dataclasses.fields(self):
            data = getattr(self, field.name)
            if field.name == "meta":
                out[field.name] = {x: torch.as_tensor(
                    y, device=device) for x, y in data.items()}  # type: ignore
            elif isinstance(data, (torch.Tensor, np.ndarray)):
                out[field.name] = torch.as_tensor(
                    data, device=device)  # type: ignore
            elif data is None:
                out[field.name] = data
            else:
                raise RuntimeError(
                    f"Not sure what to do with {field.name}: {data}")
        return EpisodeBatch(**out)
