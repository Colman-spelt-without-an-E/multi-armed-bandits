from infrastructure import *
import numpy as np
from math import log,sqrt

class UCB1_tuned(Game):
    """
    Fine-tuned upper confidence bound approach. 
    """
    def __init__(self, turns, *machines):
        super().__init__(turns,*machines)
        self.UCB_indices = [0]*self.machine_count
        self.UCB_for_var = [0]*self.machine_count

    def _update(self, index, outcome):
        super()._update(index,outcome)
        if self.next_turn >= self.machine_count + 1:
            self.UCB_for_var = [(1/len(history))*sum(reward**2 for reward in history) - self.means[i]**2 +
                            sqrt(2*log(self.next_turn)/len(history)) for i,history in enumerate(self.history)]
            
            self.UCB_indices =  [self.means[i] + sqrt(log(self.next_turn)/len(self.history[i])*min(1/4,self.UCB_for_var[i])) for i in range(self.machine_count)]

    def decide(self):
        if self.next_turn <= self.machine_count:
            return self.next_turn % self.machine_count
        return np.argmax(self.UCB_indices)
