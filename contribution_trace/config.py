from typing import Union, Optional
from collections.abc import Mapping

import torch

from .global_primitives import ModuleConfig, CompatibilityMapping
from .caching import CachingFN
from .caching.caching_fn import Qwen3CachingFN
from .tracing import TracingFN
from .scoring import ScoringFN
from .filtering import FilteringFN


HF_ID_CACHING_FN_MAPPING = {
    "Qwen/Qwen3-4B": CompatibilityMapping(
        default = Qwen3CachingFN,
        compatible= [Qwen3CachingFN]
    )
}

class TracerConfig(ModuleConfig):

    def __init__(
        self,
        model_id: str,
        caching_fn: Optional[CachingFN] = None,
        caching_strategy: str = 'all_at_once', # 'all_at_once' 'layze_tokenwise'
        attn_granularity: str = 'neuron', # 'neuron', 'head'
        caching_batch_size: int = 16,

        **kwargs
    ):
        self.model_id = model_id
        self.caching_strategy = caching_strategy
        self.attn_granualrity = attn_granularity
        self.caching_batch_size = caching_batch_size

        if not caching_fn:
            caching_fn_mapping: CompatibilityMapping = HF_ID_CACHING_FN_MAPPING.get(self.model_id, None)
            if not caching_fn_mapping:
                    raise NotImplementedError(f"There is currently no implementation for '{self.model_id}'. Please provide a custom CachingFn to run the model.")
            
            self.caching_fn = caching_fn_mapping.default()
        else:
            if not isinstance(caching_fn, CachingFN):
                raise ValueError(f"Expected 'caching_fn' to be an instance of 'CachingFN' but found '{type(caching_fn)}'.")

            elif type(caching_fn) not in caching_fn_mapping.compatible:
                raise ValueError(f"The '{type(caching_fn).__name__}' is not implemented for '{self.model_id}'.")
            
            else:
                self.caching_fn = caching_fn
        

        self.builder_config = builder_config or BuilderConfig(
            
            **builder_args
        )
        self.scorer_config = scorer_config or ScorerConfig(

            **scorer_args
        )

        super().__init__(**kwargs)
    
    @property
    def caching_manager_config(self) -> ModuleConfig:
        return ModuleConfig(
            self.device,
            self.dtype,
            model_id = self.model_id,
            caching_fn = self.caching_fn,
            strategy=self.caching_strategy,
            attn_granularity = self.attn_granualrity,
            batch_size = self.caching_batch_size
        )

    @property
    def tracing_manager_config(self) -> ModuleConfig:
        pass

    @property
    def scoring_manager_config(self) -> ModuleConfig:
        pass

    @property
    def filtering_manager_config(self) -> ModuleConfig:
        pass