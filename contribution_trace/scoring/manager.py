from typing import Optional

from ..global_primitives import Module

class ScoringManager(Module):
    pass

#     def __init__(self, config: Optional[ScorerConfig] = None, **kwargs):
#         if not config:
#             config = ScorerConfig(**kwargs)

#         self.config = config
# from typing import Union

# import torch

# from .scoring_fn import ScoringFN

# class ScorerConfig:

#     def __init__(
#         self,
#         scoring_fn: ScoringFN,
#         strategy: str = 'complete', # 'complete', 'value', 'gate'
#         batch_size: int = 16,
#         chunk_size: int = 4084,
#         device: Union[str, torch.device] = 'cuda:0',
#         **kwargs,
#     ):
#         self.scoring_fn = scoring_fn
#         self.strategy = strategy
#         self.batch_size = batch_size
#         self.chunk_size = chunk_size
#         self.device = device