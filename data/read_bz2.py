"""
How to read a .bz2 file. Use TS_large_increment.bz2 as an example. 
"""
from infrastructure import *
from TS import ThompsonSamplingBernoulli
import dill
import bz2

# read
with bz2.open('TS_large_increment.bz2', 'rb') as handle:
     ts_games = dill.load(handle)
    
# an example - inspect cumluative regret
regret = [game.historical_regret for game in ts_games]
