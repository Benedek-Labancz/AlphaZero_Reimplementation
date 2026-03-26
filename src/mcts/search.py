import torch
import numpy as np
from typing import Any
from tqdm import tqdm

from .tree import Tree

def run_simulation(tree: Tree, policy: Any, c: float) -> Tree:
    current = tree.root
    while not current.is_leaf():
        # Select the edge that maximises Q(s, a) + U(s, a)
        with torch.no_grad():
            current.update_qu_values(c=c)
            selected_edge_idx = np.argmax(current.qu_values)
            selected_edge = current.out_edges[selected_edge_idx]
        # Traverse to node where the edge leads
        if selected_edge.n == 0:
            selected_edge.add_destination_node(tree=tree) # Tree is needed to check for existing nodes and add new one if needed
        current = selected_edge.to
        # Save our path through the tree
        current.save_in_edge(selected_edge)
    if current.is_terminal():
        # If we have reached a terminal state,
        # we use the true value instead of the network estimate for backup
        v = current.env.get_winner(current.state)
    else:
        flat_representation = current.env.to_flat_representation(current.state)
        action_probs, v = policy(torch.as_tensor(flat_representation, dtype=torch.float32))
        action_probs = action_probs.squeeze()
        v = v.squeeze()
        current.expand(priors=action_probs, c=c)
    while current != tree.root:
        v = -v # Player perspectives are flipped at each level
        current.in_edge.update(v)
        current = current.in_edge.frm
    return tree

def run_mcts(tree: Tree, 
             policy: Any, 
             num_simulations: int, 
             c: float, 
             tau: float) -> tuple[Tree, np.array]:
    """
    Run simulation n times, 
    return the updated tree and the search_probabilities pi.
    """
    n = 0
    pbar = tqdm(total=num_simulations)
    while n < num_simulations:
        run_simulation(tree=tree, policy=policy, c=c)
        n += 1
        pbar.update()
    pi_values = tree.root.get_pi_values(tau=tau)
    return tree, pi_values