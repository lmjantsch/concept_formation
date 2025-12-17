from abc import ABC, abstractmethod

class ScoringFN(ABC):

    @abstractmethod
    def __call__(self):
        pass