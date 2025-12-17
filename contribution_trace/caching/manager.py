from typing import Optional, Any, Union, List, Dict, Tuple, Callable

import torch
from torch.utils.data import DataLoader
from transformers import PretrainedConfig, PreTrainedTokenizer, AutoTokenizer, AutoConfig, BatchEncoding
from nnsight import LanguageModel
import warnings

from .primitives import CachingFN, ResourceHook
from .caching_fn import CachingFN
from ..graph import NodeID
from ..constants import HF_ID_CACHING_FN_MAPPING, DEFAULT_MODEL_KWARGS

class CachingManagerConfig(ModuleConfig):

    def __init__(
        self,
        model_id: str,
        caching_fn: Optional[CachingFN] = None,
        strategy: str = 'all_at_once', # 'all_at_once' 'layze_tokenwise'
        attn_granularity: str = 'neuron', # 'neuron', 'head'
        batch_size: int = 16,
        model_args: dict = {},
        **kwargs
    ):
        self.model_id = model_id
        self.model_args = model_args

        self.strategy = strategy
        self.attn_granularity = attn_granularity
        self.batch_size = batch_size
        
        caching_fn_mapping: CompatibilityMapping = HF_ID_CACHING_FN_MAPPING.get(self.model_id, None)
        if not caching_fn_mapping:
            if not caching_fn:
                raise NotImplementedError(f"There is currently no implementation for '{self.model_id}'. Please provide a custom CachingFn to run the model.")
            self.caching_fn = caching_fn
        
        else:
            if not caching_fn:
                self.caching_fn = caching_fn_mapping.default()

            elif not isinstance(caching_fn, CachingFN):
                raise ValueError(f"Expected 'caching_fn' to be an instance of 'CachingFN' but found '{type(caching_fn)}'.")

            elif type(caching_fn) not in caching_fn_mapping.compatible:
                raise ValueError(f"The '{type(caching_fn).__name__}' is not implemented for '{self.model_id}'.")
            
            else:
                self.caching_fn = caching_fn
        
        super().__init__(**kwargs)

class CachingManager:

    def __init__(self, model_id: Optional[str] = None, caching_fn: Optional[CachingFN] = None, model: Optional[LanguageModel] = None, tokenizer: Optional[PreTrainedTokenizer] = None, model_config: Optional[PretrainedConfig] = None, config: Optional[CachingManagerConfig] = None, **kwargs):

        if not model_id and not model:
            raise ValueError("Either 'model_id' or 'model' must be defined")
        
        if model and model_id:
            warnings.warn("When providing 'model' 'model_id' is ignored")
        
        if model:
            model_id = model.name_or_path

        if not config:
            config = CachingManagerConfig(model_id, caching_fn, **kwargs)

        self.config = config
    
        self._model = model
        self._model_config = model_config
        self._tokenizer = tokenizer

        self.prompts: Optional[List[str]] = None
        self.input_ids: Optional[List[List[int]]] = None
        self.collate_fn: Optional[Callable[[List[str]], BatchEncoding]] = None

    def load(
            self, 
            inputs: Union[
                str,              # single prompt
                List[str],        # list of prompts
                List[Dict],       # single chat (sequence of messages)
                List[List[Dict]], # list of chats (batch of sequences)
            ]):
        
        if isinstance(inputs, str):
            self.prompts, self.collate_fn = self._prepare_prompt_input(inputs)
            return
        
        if isinstance(inputs, list):
            if len(inputs) == 0:
                raise ValueError("Input list cannot be empty.")
            
            first_item = inputs[0]

            if isinstance(first_item, str):
                self.prompts, self.collate_fn = self._prepare_prompt_input(inputs)
                return
            
            elif isinstance(first_item, dict):
                self.prompts, self.collate_fn = self._prepare_chat_input(inputs)
                return

            elif isinstance(first_item, list):
                self.prompts, self.collate_fn = self._prepare_chat_input(inputs)
                return
            
            raise TypeError(f"List contains unsupported element type: {type(first_item)}")

        raise TypeError(f"Expected str or List, got {type(inputs)}")
        
    def _prepare_prompt_input(self, inputs: Union[str, List[str]]) -> Tuple[List[str], Callable]:
        if isinstance(inputs, str):
            inputs = [inputs]

        if not self._tokenizer:
            self._load_tokenizer()

        prompts = self._tokenizer.batch_decode(self._tokenizer(inputs))

        def collate_fn(batch: List[str]) -> BatchEncoding:
            return self._tokenizer(batch, return_tensors='pt', padding=True)
        
        return  prompts, collate_fn

    def _prepare_chat_input(self, inputs: Union[List[Dict], List[List[Dict]]]) -> ResourceHook:
        if isinstance(inputs, dict):
            inputs = [inputs]

        if not self._tokenizer:
            self._load_tokenizer()

        prompts = self._tokenizer.apply_chat_template(inputs, tokenize=False)

        def collate_fn(batch: List[str]) -> BatchEncoding:
            return self._tokenizer.apply_chat_template(batch, return_tensors='pt', padding=True)
        
        return prompts, collate_fn

    def _load_tokenizer(self):
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(self.config.model_id)

        tokenizer.padding_side = 'right'
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        self._tokenizer = tokenizer

    def _load_config(self):
        self._model_config = AutoConfig.from_pretrained(self.config.model_id)

    def _load_model(self):
        model_kwargs = self.config.model_kwargs | DEFAULT_MODEL_KWARGS

        model = LanguageModel(
            self.config.model_id,
            config=self._model_config,
            dtype=self.config.dtype,
            device_map=self.config.device,
            **model_kwargs
        )

        self._model = model
        self._model_config = model.config

    def get_resource_hook(self) -> Callable[[], tuple]:
        return ResourceHook(self)
    
    def get_cache(self, nodes: List[NodeID]) -> Tuple[]

from typing import Union, Optional

import torch

from .caching_fn import CachingFN, Qwen3CachingFN
from ..global_primitives import CompatibilityMapping, ModuleConfig



