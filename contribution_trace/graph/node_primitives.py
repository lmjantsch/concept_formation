from dataclasses import dataclass, asdict
from typing import Union, Optional, List

import torch
import numpy as np
from numpy.typing import NDArray

COMPONENT_RANK = {
    "emb": 0,
    "pre_res": 1,
    "attn": 2,
    "mid_res": 3,
    "mlp": 4,
    "post_res": 5,
    "unembed": 6
}

# ===== Nodes =====

@dataclass(frozen=True)
class NodeID:
    """
    Base node object for contribution graph
    """
    batch_idx: int
    token_idx: int
    layer_idx: int
    component_type: str # 'attn', 'mlp', 'emb', 'target'
    node_idx: Optional[int] = None

    def __repr__(self):
        return f"Node<{self.component_type}>(L{self.layer_idx}:T{self.token_idx}:{self.node_idx})"
    
    def __lt__(self, other: 'NodeID') -> bool:
        """
        Custom comparison: less then.
        """
        if not isinstance(other, NodeID):
            return NotImplemented
        
        return self._a_before_b(self, other)
    
    def __le__(self, other: 'NodeID') -> bool:
        """
        Custom comparison: less equal.
        """
        if not isinstance(other, NodeID):
            return NotImplemented
        
        return not self._a_before_b(other, self)
    
    def __gt__(self, other: 'NodeID') -> bool:
        """
        Custom comparison: greater then.
        """
        if not isinstance(other, NodeID):
            return NotImplemented
        
        return self._a_before_b(other, self)

    def __ge__(self, other: 'NodeID') -> bool:
        """
        Custom comparison: greater equal.
        """
        if not isinstance(other, NodeID):
            return NotImplemented
        
        return not self._a_before_b(self, other)
    
    def __eq__(self, other: 'NodeID') -> bool:
        """
        Custom comparison: greater then.
        """
        if not isinstance(other, NodeID):
            return NotImplemented
        
        return self._a_equal_b(self, other)
    
    @staticmethod
    def _a_before_b(a: 'NodeID', b: 'NodeID') -> bool:
        """
        Order: Token -> Layer -> Component Rank -> Index
        Note: Implements priority lt not value lt
        """
        if a.token_idx != b.token_idx:
            return a.token_idx > b.token_idx
        
        if a.layer_idx != b.layer_idx:
            return a.layer_idx > b.layer_idx

        my_rank = COMPONENT_RANK.get(a.component_type, 99)
        other_rank = COMPONENT_RANK.get(b.component_type, 99)

        if my_rank != other_rank:
            return my_rank > other_rank
        
        return a.node_idx > b.node_idx

    @staticmethod
    def _a_equal_b(a: 'NodeID', b: 'NodeID') -> bool:
        if a.token_idx != b.token_idx:
            return False
        
        if a.layer_idx != b.layer_idx:
            return False

        my_rank = COMPONENT_RANK.get(a.component_type, 99)
        other_rank = COMPONENT_RANK.get(b.component_type, 99)

        if my_rank != other_rank:
            return False
        
        return a.node_idx == b.node_idx
    
    def is_same_module_with(self, other: 'NodeID') -> bool:
        if self.token_idx != other.token_idx:
            return False
        
        if self.layer_idx != other.layer_idx:
            return False

        return self.component_type == other.component_type

@dataclass(frozen=True)
class EmbNodeID(NodeID):
    """
    Embedding specific node object.
    """
    layer_idx: int = 0
    component_type: str = 'emb'

    def __repr__(self):
        return f"EmbNode(T{self.token_idx})"
    
@dataclass(frozen=True)
class PreResNodeID(NodeID):
    """
    PreRes specific node object.
    """
    component_type: str = 'mlp'

    def __repr__(self):
        return f"PreResNode(L{self.layer_idx}:T{self.token_idx})"

@dataclass(frozen=True)
class AttnNodeID(NodeID):
    """
    Attention specific node object.
    """
    key_idx: int
    component_type: str = 'attn'

    def __repr__(self):
        return f"AttnNod(L{self.layer_idx}:T{self.token_idx}:K{self.key_idx}:{self.node_idx})"
    
    @staticmethod
    def _a_before_b(a: 'NodeID', b: 'NodeID') -> bool:
        """
        Order: Token -> Layer -> Component Rank -> Index
        """
        if a.token_idx != b.token_idx:
            return a.token_idx < b.token_idx
        
        if a.layer_idx != b.layer_idx:
            return a.layer_idx < b.layer_idx

        my_rank = COMPONENT_RANK.get(a.component_type, 99)
        other_rank = COMPONENT_RANK.get(b.component_type, 99)

        if my_rank != other_rank:
            return my_rank < other_rank
        
        if a.key_idx != b.key_idx:
            return a.key_idx < b.key_idx
        
        return a.node_idx < b.node_idx

    @staticmethod
    def _a_equal_b(a, b: 'NodeID') -> bool:
        if a.token_idx != b.token_idx:
            return False
        
        if a.layer_idx != b.layer_idx:
            return False

        my_rank = COMPONENT_RANK.get(a.component_type, 99)
        other_rank = COMPONENT_RANK.get(b.component_type, 99)

        if my_rank != other_rank:
            return False
        
        if a.key_idx != b.key_idx:
            return False
        
        return a.node_idx == b.node_idx
    
@dataclass(frozen=True)
class MidResNodeID(NodeID):
    """
    PreRes specific node object.
    """
    component_type: str = 'mlp'

    def __repr__(self):
        return f"MidResNode(L{self.layer_idx}:T{self.token_idx})"

@dataclass(frozen=True)
class MLPNodeID(NodeID):
    """
    MLP specific node object.
    """
    component_type: str = 'mlp'

    def __repr__(self):
        return f"MLPNode(L{self.layer_idx}:T{self.token_idx}:{self.node_idx})"
    
@dataclass(frozen=True)
class PostResNodeID(NodeID):
    """
    PreRes specific node object.
    """
    component_type: str = 'mlp'

    def __repr__(self):
        return f"PostResNode(L{self.layer_idx}:T{self.token_idx})"

@dataclass(frozen=True)
class UnEmbNodeID(NodeID):
    """
    PreRes specific node object.
    """
    layer_idx: int = 9999
    component_type: str = 'mlp'

    def __repr__(self):
        return f"UnEmbNode(T{self.token_idx})"


# ===== Edges =====

@dataclass(frozen=True)
class Edge:
    source: NodeID
    target: NodeID
    score: float
    edge_type: str # 'gate_edge', 'value_edge'


# ===== Node Wrapper =====

@dataclass(frozen=True)
class WrappedNode:
    nodes: Union[List[NodeID], List[List[NodeID]]]

    def __getattr__(self, name):
        """
        Delegate attributes to the wrapped node objected
        """
        return getattr(self.node, name)

    def __repr__(self):
        return f"WrappedNode({repr(self.node)})"

@dataclass(frozen=True)
class TargetNode:
    def __repr__(self):
        return f"TargetNode({repr(self.node)})"

   
# ===== Node Lists =====

@dataclass(frozen=True)
class NodeList:
    nodes: Union[List[NodeID], List[List[NodeID]]]

    def __post_init__(self):
        for key, value in asdict(self).items():
            if self.size(attr=key) != self.size():
                raise ValueError(f"Expected all members to be of size {self.size()}, but got {self.size(attr=key)} for {key}.")

    def __getitem__(self, key: Union[int, str, slice, List[str], List[int]]) -> Union['NodeList', WrappedNode, List]:
        # String (Column Selection)
        if isinstance(key, str):
            if not hasattr(self, key):
                raise KeyError(f"Column '{key}' not found.")
            return getattr(self, key)
        
        # Int (Row Selection)
        if isinstance(key, int):
            if key < 0:
                key += self.length
            if key < 0 or key >= self.length:
                raise IndexError("Row index out of range")
            return WrappedNode(**({'node': self.nodes[key]} | {key: value[key] for key, value in asdict(self).items() if key != 'nodes'}))

        # Slice (Row Selection) -> returns DataDict
        if isinstance(key, slice):
            return type(self)({k:v[key] for k, v in asdict(self).items()})
            
        raise TypeError(f"Unsupported key type: {type(key)}")

    def size(self, dim: Optional[int] = None, attr: Optional[str] = None):

        if attr:
            obj = getattr(self, attr)
        else:
            obj = self.nodes()
        
        if isinstance(object, torch.Tensor):
            return obj.size(dim)
        
        if isinstance(obj, NDArray):
            return obj.size(dim)
        
        size = []
        while isinstance(obj, List):
            size.append(len(obj))
            obj = obj[0]

        if dim:
            dim_range = list(range(-len(size), len(size)))

            if dim in dim_range:
                raise ValueError(f"Expected 'dim' to be in {dim_range}, but found {dim}")
            
            return size[dim]
        
        return size

    def __len__(self):
        self.size(0)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.nodes[0])})"
    
@dataclass
class NextTargetNodeList(NodeList):
    total_contributions: Optional[torch.Tensor] = None
    value_contributions: Optional[torch.Tensor] = None
    gate_contributions: Optional[torch.Tensor] = None

    gate_activation: Optional[torch.Tensor] = None
    value_activation: Optional[torch.Tensor] = None
    layer_norm_std: Optional[torch.Tensor] = None
    
@dataclass
class ContributionPartsList(NodeList):
    part_vectors: torch.Tensor

@dataclass
class TargetNodeList(NodeList):
    target_vectors: torch.Tensor