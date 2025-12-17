import networkx as nx
from typing import List, Tuple, Dict, Set, Union

import torch

from .node_primitives import NodeID, NextTargetNodeList, Edge
from .priority_queue import PriorityQueue


class ContributionGraph:
    #TODO support batching

    def __init__(self):
        self._graph = nx.DiGraph()
        self._priority_queue = PriorityQueue()

    def add_edge(self, edge: Edge):
        if edge.target not in self._priority_queue:
            self._priority_queue.push(edge.target)

        self._graph.add_edge(
            edge.source,
            edge.target,
            weight=edge.score,
            edge_type=edge.edge_type
        )
    def add_edges(self, edges: List[Edge]):
        for edge in edges:
            self.add_edge(edge)
    
    def add_node(self, node: NodeID):
        if node not in self._priority_queue:
            self._priority_queue.push(node)

        self._graph.add_node(node)

    def add_nodes(self, nodes: List[NodeID]):
        for node in nodes:
            self.add_node(node)
    
    def get_next_nodes(self) -> NextTargetNodeList:
        if self._priority_queue.is_empty():
            return NextTargetNodeList([])
        
        first_node = self._priority_queue.pop()

        nodes = []
        contributions = []
        while True:
            if not first_node.is_same_module_with(self._priority_queue.peek()):
                break
            node = self._priority_queue.pop()
            nodes.append(node)
            total_sum, (value_sum, gate_sum) = self._aggregate_in_edges(node)
            contributions.append((total_sum, value_sum, gate_sum))

        total_contributions, value_contributions, gate_contributions = torch.Tensor(contributions).transpose(0, 1)
        
        return NextTargetNodeList(
            nodes, 
            total_contributions=total_contributions, 
            value_contributions=value_contributions,
            gate_contributions=gate_contributions
        )

    def _aggregate_in_edges(self, node: NodeID) -> Tuple[float, Tuple[float, float]]:
        value_sum = 0.0
        gate_sum = 0.0
        
        for _, _, data in self._graph.in_edges(node, data=True):
            if data["edge_type"] == "value":
                value_sum += data["score"]
            elif data["edge_type"] == "gate":
                gate_sum += data["score"]
        
        total_sum = gate_sum + value_sum           
        return total_sum, (value_sum, gate_sum)
    
    def has_unvisited_nodes(self):
        return not self._priority_queue.is_empty()
    
    @property
    def nx_graph(self) -> nx.DiGraph:
        return self._graph
    
    def __len__(self): 
        return self._graph.number_of_nodes()

    def __contains__(self, node: NodeID):
        return node in self._graph