import numpy as np

from src.mcts.tree import Tree, Node
from src.mcts.search import run_mcts


def select_az_action(rng, tree: Tree, policy, num_simulations: int, c: float, **kwargs):
    '''
    Run N simulations then select the action with the highest search probability.
    '''
    _, pi_values = run_mcts(
        rng=rng,
        tree=tree,
        policy=policy,
        num_simulations=num_simulations,
        c=c,
        tau=1 # we do not care about this, as we're doing argmax
    )
    print(pi_values)
    max_visit_count_idx = np.argmax(pi_values)
    action = tree.root.valid_actions[max_visit_count_idx]
    return action


def play_episodes(rng,
                  env, 
                    best_policy, 
                    candidate_policy,
                    best_select_action_fn: callable,
                    candidate_select_action_fn: callable,
                    num_games: int, 
                    num_simulations: int,
                    c: float,
                    **kwargs):
    '''
    Play an episode of the provided environment using
    two (potentially different) policies.
    Two search trees are maintained in parallel.

    
    '''
    num_wins = [0, 0]
    policies = [best_policy, candidate_policy]
    select_action_fns = [best_select_action_fn, candidate_select_action_fn]
    for i in range(num_games):
        initial_state, _ = env.reset()
        trees = [
            Tree(root=Node(env=env, state=initial_state)),
            Tree(root=Node(env=env, state=initial_state)),
        ]
        # 0 or 1 to index policies and trees (0=best, 1=candidate)
        starting_player = round(rng.random())
        current_player = starting_player
        done = False
        num_steps = 0
        while not done:
            env.render() # Allow for user interaction; ignored if env.render_mode == None
            # Select the action based on the current player's policy
            action = select_action_fns[current_player](
                rng=rng,
                tree=trees[current_player],
                policy=policies[current_player],
                num_simulations=num_simulations,
                c=c,
                **kwargs
            )
            state, _, terminated, truncated, _ = env.step(action)
            # Traverse the selected edge on both trees
            for i, tree in enumerate(trees):
                if tree.root.is_leaf():
                    # We can expand this node without priors, as it is going to be discarded
                    tree.root.expand()
                traversed_edge = tree.root.get_out_edge_by_action(action)
                if traversed_edge.n == 0:
                    traversed_edge.add_destination_node(tree=tree)
                trees[i] = tree.switch_root(node=traversed_edge.to) # switch root and retain the search tree
                assert np.array_equal(trees[i].root.state, state)
            # Switch players
            current_player = 1 - current_player
            num_steps += 1
            done = (terminated or truncated)
        winner = env.get_winner(state)
        if winner == 1:
            num_wins[current_player] += 1
        elif winner == -1:
            num_wins[1 - current_player] += 1
    env.close()
    return num_wins