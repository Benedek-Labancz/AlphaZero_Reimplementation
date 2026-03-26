import numpy as np
import torch
from torch.distributions.dirichlet import Dirichlet

from src.mcts.tree import Tree
from src.mcts.search import run_mcts


def play_episode(tree: Tree, 
                 policy,
                 num_simulations: int, 
                 c: float,
                 tau: float=1.0,
                 early_selection_threshold: int=30,
                 epsilon: float=0.25,
                 noise_concentration: float=0.03):
    ep_states = []
    ep_pi_values = []
    ep_winners = []
    current = tree.root
    move_count = 0
    while not current.is_terminal():
        ep_states.append(current.state.copy())
        if current.priors is not None:
            # Sample Dirichlet noise and add it to root priors
            dir_dist = Dirichlet(torch.tensor(len(current.priors) * [noise_concentration]))
            current.priors = (1 - epsilon) * current.priors + epsilon * dir_dist.sample()
        tree, pi_values = run_mcts(
            tree=tree,
            policy=policy,
            num_simulations=num_simulations,
            c=c,
            tau=tau
        )
        ep_pi_values.append(pi_values)
        # Early in the game, we use pi values to select actions
        if move_count < early_selection_threshold:
            selected_edge = np.random.choice(current.out_edges, p=pi_values)
        else:
            # Later on we select greedily, i.e. tau -> 0
            selected_edge_idx = np.argmax(pi_values)
            selected_edge = current.out_edges[selected_edge_idx]
        if selected_edge.n == 0:
            selected_edge.add_destination_node(tree=tree)
        # Set new node as root, discard the rest of the tree
        current = selected_edge.to
        tree = Tree(root=current)
    # Get the actual winner, and back up the values for supervised learning examples
    # The sign alternates as player roles alternate
    r = current.env.get_winner(current.state)
    for _ in range(len(ep_states)):
        ep_winners.insert(0, r)
        r = -1 * r
    return ep_states, ep_pi_values, ep_winners

        
