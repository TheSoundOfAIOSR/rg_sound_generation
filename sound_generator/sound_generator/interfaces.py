from abc import abstractmethod
from typing import Dict


class DataProcessor:
    @abstractmethod
    def process(self, inputs: Dict) -> Dict:
        ...


class BaseModel:
    @abstractmethod
    def _load_model(self):
        ...

    @abstractmethod
    def predict(self, inputs: Dict) -> Dict:
        ...
