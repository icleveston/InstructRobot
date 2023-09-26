from abc import ABC, abstractmethod


class Conf(ABC):

    def __init__(self, name: str, scene_file: str, instruction_set: ()):
        self._name = name
        self._scene_file = scene_file
        self._instruction_set = instruction_set

    def __str__(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(self._instruction_set)

    def __getitem__(self, position) -> ():
        return self._instruction_set[position]

    @abstractmethod
    def configure(self):
        pass

    @property
    def scene_file(self):
        return self._scene_file

    @property
    def instruction_set(self):
        return self._instruction_set
