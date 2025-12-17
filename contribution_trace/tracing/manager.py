from typing import Optional, Union, List

import warnings

from ..graph import NodeID, TargetNodeList, NodeList, UnEmbNodeID
from ..caching import ResourceHook
from ..global_primitives import Module

class TracingManager(Module):
    pass

#     def __init__(
#         self,
#         target_layer: int,
#         target_location: str, # 'unemb', 'pre_res', 'mid_res', 'post_res'
#         target_type: str, # logit, '
#         tracing_fn: TargetBuilderFN,
#         trace_builder_fn: TraceBuilderFN,
#         batch_size: int = 16,
#         device: Union[str, torch.device] = 'cuda:0',):
#         if not config:
#             config = BuilderConfig(**kwargs)

#         self.config = config
#         self.target_nodes = None

#     def load(
#         self, 
#         resource_hook: ResourceHook,
#         targets: Optional[TargetNodeList]
#         ):
#         self.resource_hook = resource_hook

#         if targets and self.config.target_build_fn:
#             warnings.warn("When 'TargetNodeList' is provided, the 'target_build_fn' is ignored.")
#         self.targets = targets

#     def get_init_nodes(self) -> List[NodeID]:
        

#         from typing import Union

# import torch

# class BuilderConfig:

#     def __init__(
#         self,
#         target_layer: int,
#         target_location: str, # 'unemb', 'pre_res', 'mid_res', 'post_res'
#         target_type: str, # logit, '
#         target_build_fn: TargetBuilderFN,
#         trace_builder_fn: TraceBuilderFN,
#         batch_size: int = 16,
#         device: Union[str, torch.device] = 'cuda:0',
#         **kwargs
#     ):
#         self.target_build_fn = target_build_fn
#         self.trace_build_fn = trace_builder_fn
#         self.batch_size = batch_size
#         self.device = device