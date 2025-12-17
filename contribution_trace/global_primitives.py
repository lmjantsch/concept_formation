from dataclasses import dataclass
from collections.abc import Mapping
from typing import Type, List, Union, Optional

import torch

@dataclass(frozen=True)
class CompatibilityMapping:
    default: Type
    compatible: List[Type]

class Module:

    def __init__(self):
        self.config = ModuleConfig()

    def to(self, *args, device: Optional[Union[str, torch.device]] = None, dtype: Optional[torch.dtype] = None, **kwargs):
        for arg in args:
            if isinstance(arg, torch.device):
                self.config.device = arg
            elif isinstance(arg, torch.dtype):
                self.config.dtype = arg
            elif isinstance(arg, str):
                try:
                    self.config.device = torch.device(arg)
                except RuntimeError as e:
                    raise ValueError(f"Invalid device string: '{arg}'") from e
            else:
                raise ValueError(f"Can't assign the module to '{arg}' (type: {type(arg).__name__}).")

        if device:
            if isinstance(device, torch.device):
                self.config.device = device
            else:
                self.config.device = torch.device(device)

        if dtype:
            self.config.dtype = dtype

class Config(Mapping):

    def __init__(self, kwargs):
        self.kwargs = kwargs

    def __getitem__(self, key):
        if key in self.kwargs:
            return self.kwargs[key]
        return self.__dict__[key]
    
    def __iter__(self):
        for key in self._flat_keys:
            yield key

    def __len__(self):
        return len(self._flat_keys)
    
    @property
    def _flat_keys(self):
        return [k for k in self.__dict__ if k != 'kwargs'] + [*self.kwargs]
    
class ModuleConfig(Module):

    def __init__(self, device: Optional[Union[str, torch.device]] = 'cpu', dtype: Optional[torch.dtype] = torch.bfloat16, **kwargs):
        self.device = device
        self.dtype = dtype
        super().__init__(**kwargs)