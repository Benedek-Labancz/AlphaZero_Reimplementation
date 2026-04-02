import torch
from torch.nn.functional import softmax
import numpy as np
from typing import Any

from .tree import Tree

def run_simulation(rng, tree: Tree, policy: Any, c: float) -> Tree:
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
        with torch.no_grad():
            # Add rotation and flipping in a random manner
            transformed_state, transformation = current.env.get_random_transformed_state(rng=rng, state=current.state)
            flat_representation = current.env.to_flat_representation(transformed_state)
            # TODO: if running multiple instances of MCTS, add state to queue and evaluate in batches
            transformed_action_logits, v = policy(torch.as_tensor(flat_representation, dtype=torch.float32))
            transformed_action_logits = transformed_action_logits.squeeze()
            # turn logits into probabilities
            transformed_action_probs = softmax(transformed_action_logits, dim=0)
            # perform the inverse transformation on the action probs so that the predictions are lined up with the actual state
            action_probs = current.env.get_original_action_probs_from_transformed(transformed_action_probs, transformation)
            assert np.array_equal(
                current.env._transform_board(
                action_probs.reshape(current.env.get_num_dimensions() * (current.env.get_board_size(),)), 
                **transformation
                ).reshape(-1),
                transformed_action_probs
            )
            v = v.squeeze()
            current.expand(priors=action_probs)
    while current != tree.root:
        v = -v # Player perspectives are flipped at each level
        current.in_edge.update(v)
        current = current.in_edge.frm
    return tree

def run_mcts(rng,
             tree: Tree, 
             policy: Any, 
             num_simulations: int, 
             c: float, 
             tau: float) -> tuple[Tree, np.array]:
    """
    Run simulation n times, 
    return the updated tree and the search_probabilities pi.
    """
    n = 0
    while n < num_simulations:
        run_simulation(rng=rng, tree=tree, policy=policy, c=c)
        n += 1
    pi_values = tree.root.get_pi_values(tau=tau)
    return tree, pi_values