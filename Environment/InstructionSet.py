
class InstructionSet:

    def __init__(self, name: str, instruction_set: ()):
        self._name = name
        self._instruction_set = instruction_set

    def __str__(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(self._instruction_set)

    def __getitem__(self, position) -> ():
        return self._instruction_set[position]
