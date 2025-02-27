# robust_cbf

This code simulates a multi-agent environment in which a controlled agent (blue) navigates from a start to goal position
while avoiding collisions with other agents (red). Each of the other agents has a randomized (unknown) goal that they
try to reach, but approximately half of them blindly travel from their start to goal position, while the other half
exhibit some collision avoidance behavior. Our code implements the algorithm in the paper "Safe Multi-Agent Interaction
through Robust Control Barrier Functions with Learned Uncertainties", which maintains safety of the blue agent by using
a robust multi-agent Control Barrier Function (CBF) that predicts and accounts for uncertainty about other agents(red).

Run "python game_GP.py" to run the simulation with the robust multi-agent Control Barrier Function (CBF) in a randomized
environment. This runs 1000 trials, first with the robust CBF controller, and next with the nominal CBF controller. The
CBF code setting up the QP is found in "control.py". The MVG code predicting the uncertainties is found in "
GP_predict.py"

The script "GP_train.py" optimizes the MVG hyperparameters, based on entered training data. To execute it, you must load
in a set of training data.

# TODO

- [x] gymnasium api for game simualtor
- [x] scale rendering window (zoom level)
- [x] test consistency of simulation results
- [x] identify cbf tunable parameters in paper and code
- [ ] collect simulations under different cbf parameters, statistics like success rate, average time, collision rate...
- [ ] collect simulations under different cbf parameters, with different fixed nr of agents
- [ ] plot statistics for various cbf parameters (succ rate, collision rate, ...)
- [ ] create (N, N, 3) observation space as grid (N, N) with 3-channels for positions, velocities, and roles


