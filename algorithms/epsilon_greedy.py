from infrastructure import *
import numpy as np
from random import choices

class epsilon_greedy(Game):
    """
    epsilon-greedy approach.
    """
    def __init__(self, turns, *machines, epsilon=0.5):
        super().__init__(turns, *machines)
        self.epsilon = epsilon

    def decide(self):
        if self.next_turn <= self.machine_count:
            return self.next_turn % self.machine_count
        exploit = choices([0,1],[self.epsilon,1-self.epsilon])[0]
        if exploit == 1: 
            return np.argmax(self.means)
        return choices(range(self.machine_count))[0]

    def _update(self,index,outcome):
        super()._update(index,outcome)
