from abc import ABC, abstractmethod


class BaseLoader(ABC):
    def __init__(self, data_path):
        self.data_path = data_path

    @abstractmethod
    def load(self):
        ...

    @abstractmethod
    def skeleton_model(self):
        ...
