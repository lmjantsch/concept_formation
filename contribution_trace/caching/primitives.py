from abc import ABC, abstractmethod

from .manager import CachingManager

class CachingFN(ABC):

    @abstractmethod
    def __call__(self):
        pass

class ResourceHook:
    def __init__(self, manager: 'CachingManager'):
        self._manager = manager
    
    @property
    def model(self):
        if not self._manager._model:
            self._manager._load_model()
        return self._manager._model
    
    @property
    def tokenizer(self):
        if not self._manager._tokenizer:
            self._manager._load_tokenizer()
        return self._manager._tokenizer
        
    @property
    def config(self):
        if not self._manager._model_config:
            self._manager._load_config()
        return self._manager._model_config
    
    @property
    def input_ids(self):
        return self._manager.input_ids
    
    @property
    def input_ids(self):
        return self._manager.input_ids