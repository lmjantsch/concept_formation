from abc import ABC, abstractmethod

class FilteringFN(ABC):

    @abstractmethod
    def __call__(self):
        pass