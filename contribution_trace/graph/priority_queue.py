import heapq
from typing import Union, List, Optional

from .node_primitives import NodeID

class PriorityQueue:
    #TODO support batching
    
    def __init__(self, data: Optional[Union[NodeID, List[NodeID]]] = None):
        self._heap = [] 
        self._heap_elements = set()
        
        if data:
            if not isinstance(data, List):
                data = [data]
            for item in data:
                self.push(item)

    def push(self, item: NodeID) -> None:
        if not isinstance(item, NodeID):
            raise TypeError(f"Item must be of type NodeID, got {type(data).__name__}")
        
        heapq.heappush(self._heap, item)
        self._heap_elements.add(item)

    def pop(self) -> NodeID:
        if self.is_empty:
            raise IndexError("pop from empty PriorityQueue")
        
        item = heapq.heappop(self._heap)
        self._heap_elements.remove(item)
        return item

    def peek(self) -> NodeID:
        if self.is_empty:
            raise IndexError("peek from empty PriorityQueue")
        
        return self._heap[0]
    
    def is_empty(self) -> bool:
        return len(self._heap) == 0

    def __len__(self) -> int:
        return len(self._heap)
    
    def __contains__(self, node: NodeID):
        return node in self._heap_elements