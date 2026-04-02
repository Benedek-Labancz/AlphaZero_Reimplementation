import numpy as np

from src.mcts.tree import Tree

def select_random_action(rng, tree: Tree, **kwargs):
    if tree.root.is_leaf():
        tree.root.expand()
    num_valid_actions = len(tree.root.valid_actions)
    return tree.root.valid_actions[rng.integers(num_valid_actions)]