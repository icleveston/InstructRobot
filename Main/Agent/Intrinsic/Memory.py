
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.states_intrinsic = []
        self.intrinsic_states = []
        self.logprobs = []
        self.rewards_ext = []
        self.rewards_int = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.states_intrinsic[:]
        del self.intrinsic_states[:]
        del self.logprobs[:]
        del self.rewards_ext[:]
        del self.rewards_int[:]
        del self.is_terminals[:]

    @property
    def rewards(self):
        return self.rewards_ext
