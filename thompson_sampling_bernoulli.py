from infrastructure import *

"""
Suppose we are given 3 machines with payouts following Bern(0.33), Bern(0.55)
and Bern(0.6) respectively and we play 1000 rounds in total.
"""

# initialise machines
machines = [bernoulli_machine(i) for i in [0.33, 0.55, 0.6]]

# α, β parameters for beta prior
# α = β = 1 gives uniform distribution
priors_ab = [[1,1] for i in range(len(machines))]

class ThompsonSamplingBernoulli(Game):
    # add posterior parameters
    def __init__(self, prior_parameters, turns, *machines):
        super().__init__(turns, *machines)
        self.prior_parameters = prior_parameters
        self.post_parameters = [[] for i in range(len(machines))]
        
    # overwrite decide
    def decide(self):
        # beta posterior distribution parameters
        post_samples = []
        for k, hist in enumerate(self.history):
            N = sum(hist)
            a = N+self.prior_parameters[k][0]
            b = len(hist)-N+self.prior_parameters[k][1]
            post_samples.append(np.random.beta(a, b))
            self.post_parameters[k].append([a, b])
        return np.argmax(post_samples)

g = ThompsonSamplingBernoulli(priors_ab, 1000, *machines)
g.simulate()

# visualisation of posterior distributions
from scipy.stats import beta
import matplotlib.pyplot as plt

def plot_beta_pdf(ax, a, b):
    x = np.linspace(0.01, 0.99, 99)
    ax.plot(x, beta(a, b).pdf(x))

turns = [0,1,2,4,9,99,249,499, 999]
fig, ax = plt.subplots(len(turns), figsize=(5,30))
for n, t in enumerate(turns):
    for k in range(len(machines)):
        a, b = g.post_parameters[k][t]
        plot_beta_pdf(ax[n], a, b,)
        ax[n].legend([i for i in range(1,4)])
    ax[n].set_title(f"Posterior distributions at turn {t+1}")