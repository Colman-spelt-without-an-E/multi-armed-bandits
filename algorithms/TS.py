from infrastructure import *
from copy import deepcopy

class ThompsonSamplingBernoulli(Game):
    """
    Thompson sampling using beta prior.
    
    Parameters
    ----------
        prior_parameters : list
            Nested list containing the beta prior parameters for each machine
            e.g. [[1,1], [1,1], [1,1]]
        turns : int
            Number of turns to be played
        *machines : list
            List of Machines
    """
    
    # add posterior parameters
    def __init__(self, prior_parameters, turns, *machines):
        """
        Constructs attributes. 
        
        Attributes
        ----------
            post_parameters : list
                Nested list containing the beta posterior parameters for each machine after each turn
            post_parameters_history : list
                Stores the beta posterior parameters for each machine and each turn
        """
        super().__init__(turns, *machines)
        self.post_parameters = deepcopy(prior_parameters)
        self.post_parameters_history = [deepcopy(prior_parameters)]
        
    # overwrite decide
    def decide(self):
        
        # beta posterior distribution parameters
        post_samples = [np.random.beta(a, b) for a, b in self.post_parameters]
        return np.argmax(post_samples)
    
    # overwrite _update to store posterior parameters
    def _update(self, index, outcome):
        super()._update(index, outcome)
        
        # update the posterior distribution at given index
        if outcome == 1:
            self.post_parameters[index][0] += 1
        else:
            self.post_parameters[index][1] += 1
        self.post_parameters_history.append(deepcopy(self.post_parameters))
