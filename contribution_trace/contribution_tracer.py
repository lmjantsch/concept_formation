from typing import List, Any, Optional, Union, Dict

import warnings
import networkx as nx

from .graph import ContributionGraph
from .caching import CachingManager, CachingFN
from .tracing import TracingManager, TracingFN
from .scoring import ScoringManager, ScoringFN
from .filtering import FilteringManager, FilteringFN
from .config import TracerConfig
from .global_primitives import Module

class ContributionTracer(Module):

    def __init__(self, model_id: Optional[str] = None, tracing_fn: Optional[TracingFN] = None, scoring_fn: Optional[ScoringFN] = None ,filtering_fn:Optional[FilteringFN] = None, caching_fn: Optional[type] = None, config: Optional[TracerConfig] = None, **kwargs):
        if config and any[model_id, tracing_fn, scoring_fn, caching_fn, filtering_fn]:
            warnings.warn("When a initiated TracerConfig is provided, other configurations are ignored.")

        if not config:
            config = TracerConfig(model_id, tracing_fn, scoring_fn, caching_fn, filtering_fn, **kwargs)
        self.config = config
        self._graph = ContributionGraph()

        self._initialize_manager()

    def _initialize_manager(self):
        self._caching_manager = CachingManager(**self.config.caching_manager_config)

        self._tracing_manager = TracingManager(**self.config.tracing_manager_config)

        self._scoring_manager = ScoringManager(**self.config.scoring_manager_config)

        self._filtering_manager = FilteringManager(**self.config.filtering_manager_config)

    def trace(self, inputs: Any, targets: Optional[Any] = None):
        resource_hook = self._cache_manager.load(inputs)
        self._tracing_manager.load(resource_hook, targets)
        
        init_nodes = self._tracing_manager.get_init_nodes()
        self._graph.add_node(init_nodes)

        while not self._graph.has_unvisited_nodes():
            next_target_nodes = self._graph.get_next_nodes()

            next_target_nodes, cached_contribution_parts = self._caching_manager.get_cache(next_target_nodes)

            next_target_nodes = self._tracing_manager.build_targets(next_target_nodes)

            scored_contributon_parts = self._scoring_manager.score(next_target_nodes, cached_contribution_parts)

            filtered_contribution_parts = self._filtering_manager.select(scored_contributon_parts)

            contribution_edges = self._get_contribution_edges(next_target_nodes, filtered_contribution_parts)

            self._graph.add_edge(contribution_edges)

    def _get_contribution_edges():
        pass

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph.nx_graph

    def save(self, path: str):
        # TODO
        pass

