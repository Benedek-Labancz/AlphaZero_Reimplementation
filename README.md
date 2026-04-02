# AlphaZero

This is a reimplementation of the AlphaZero algorithm as described by [Silver et al. (2017)](https://www.nature.com/articles/nature24270).

We use PyTorch and Numpy as the primary backend.

The algorithm is applied to a custom game of repeated multi-dimensional TicTacToe, 
but could be easily adapted to play any other two-player perfect information game.


### Main Features

1. PyTorch reimplementation of MCTS and policy training from scratch
2. Custom game environment following the Gym API
3. Parallel training workflow using Python's `multiprocessing` module
