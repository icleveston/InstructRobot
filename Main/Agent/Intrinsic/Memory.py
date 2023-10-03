
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.intrinsic_states = []
        self.logprobs = []
        self.rewards_ext = []
        self.rewards_int = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.intrinsic_states[:]
        del self.logprobs[:]
        del self.rewards_ext[:]
        del self.rewards_int[:]
        del self.is_terminals[:]
