# Multi-armed Bandits

The multi-armed bandit problem is a problem in probability theory where limited resources are allocated between alternative (competing) projects. We will be focusing on the problem where there is a fixed number of slot machines, each with payoffs following fixed but unknown distributions, and there is a fixed number of turns, or horizon, to play the slot machines. The goal is to maximise the profit. The set-up is represented as a class in Python called "Game", and is stored in the file infrastructure.py. 

In this repository, 5 strategies are used. They are $\epsilon$-greedy, upper confidence bound (UCB), greedy Bayesian, Thompson sampling, and randomised probability matching (RPM). Each strategy is stored as a class in a Python file, under the folder "algorithms". 

Comparisons of strategies are also made under the folder "analysis". 

For details, please refer to M2R_Stats.pdf. In the pdf, non-stochastic gambling and continuum-armed bandit problems are explored as well. 

This is written in collaboration with Afshein Keshmiri, Justin Lau, Linze Li, Lunzhi Shi, and Lucas Siu. See also Justin's repository https://github.com/n88k/multi-armed-bandit. 
