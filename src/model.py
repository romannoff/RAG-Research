from abc import ABC, abstractmethod
from typing import TypedDict

class ModelAnswer(TypedDict):
    query: str
    context: list[str]
    answer: str

class Model(ABC):
    
    @abstractmethod
    def __call__(self, *args, **kwds) -> ModelAnswer: pass