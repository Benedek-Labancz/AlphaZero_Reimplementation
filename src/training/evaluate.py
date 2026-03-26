import numpy as np

from src.mcts.tree import Tree, Node
from src.mcts.search import run_mcts


def select_action(tree: Tree, policy, num_simulations: int, c: float):
    _, pi_values = run_mcts(
        tree=tree,
        policy=policy,
        num_simulations=num_simulations,
        c=c,
        tau=1 # we do not care about this, as we're doing argmax
    )
    max_visit_count_idx = np.argmax(pi_values)
    action = tree.root.valid_actions[max_visit_count_idx]
    return action


def evaluate_policy(env, 
                    best_policy, 
                    candidate_policy, 
                    num_games: int, 
                    num_simulations: int,
                    c: float):
    num_wins = 0
    policies = [best_policy, candidate_policy]
    for i in range(num_games):
        initial_state, _ = env.reset()
        trees = [
            Tree(root=Node(env=env, state=initial_state)),
            Tree(root=Node(env=env, state=initial_state)),
        ]
        # 0 or 1 to index policies and trees (0=best, 1=candidate)
        starting_player = round(np.random.random())
        current_player = starting_player
        done = False
        num_steps = 0
        while not done:
            # Select the action based on the current player's policy
            action = select_action(
                tree=trees[current_player],
                policy=policies[current_player],
                num_simulations=num_simulations,
                c=c
            )
            state, _, terminated, truncated, _ = env.step(action)
            # Traverse the selected edge on both trees
            for i, tree in enumerate(trees):
                if tree.root.is_leaf():
                    # We can expand this node with dummy priors, as it is going to be discarded
                    tree.root.expand(priors=np.zeros(tree.root.env.total_num_actions))
                traversed_edge = tree.root.get_out_edge_by_action(action)
                if traversed_edge.n == 0:
                    traversed_edge.add_destination_node(tree=tree)
                trees[i] = Tree(root=traversed_edge.to)
                # assert np.array_equal(trees[i].root.state, state)
            # Switch players
            current_player = 1 - current_player
            num_steps += 1
            done = (terminated or truncated)
        winner = env.get_winner(state)
        if current_player == 0 and winner == -1:
            # Best policy lost
            num_wins += 1
        elif current_player == 1 and winner == 1:
            # Candidate policy won
            num_wins += 1
    env.close()
    return num_wins / num_games