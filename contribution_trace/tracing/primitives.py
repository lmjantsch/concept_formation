from abc import ABC, abstractmethod

class TracingFN(ABC):

    @abstractmethod
    def __call__(self):
        pass