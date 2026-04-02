import numpy as np

from src.mcts.tree import Tree, Node


def select_minimax_action(rng, tree: Tree, max_depth: int, epsilon: float=0, **kwargs):
    if tree.root.is_leaf():
        tree.root.expand()
    if rng.random() < epsilon:
        num_valid_actions = len(tree.root.valid_actions)
        return tree.root.valid_actions[rng.randint(num_valid_actions)]
    else:
        alpha = -np.inf
        beta = np.inf
        selected_action = None
        for edge in tree.root.out_edges:
            if edge.n == 0:
                    edge.add_destination_node(tree=tree)
            minimax_v = get_minimax_value(
                tree=Tree(root=edge.to, nodes=tree.nodes.copy()),
                max_depth=max_depth,
                depth=1,
                alpha=alpha,
                beta=beta
            )
            if minimax_v > alpha:
                selected_action = edge.action
                alpha = minimax_v
        return selected_action

def get_minimax_value(tree: Tree, max_depth: int, depth: int, alpha: int, beta: int):
    if tree.root.is_terminal() or depth == max_depth:
        # print(get_leaf_value(node=tree.root))
        return get_leaf_value(node=tree.root, depth=depth)
    if tree.root.is_leaf():
        tree.root.expand()
    minimax_values = []
    for edge in tree.root.out_edges:
        if edge.n == 0:
            edge.add_destination_node(tree=tree)
        minimax_v = get_minimax_value(
            tree=Tree(root=edge.to, nodes=tree.nodes.copy()),
            max_depth=max_depth,
            depth=depth+1,
            alpha=alpha,
            beta=beta
        )
        minimax_values.append(minimax_v)
        if depth % 2 == 0: # Maximising player
            if minimax_v > alpha:
                alpha = minimax_v
        else: # Minimising player
            if minimax_v < beta:
                beta = minimax_v
        if alpha >= beta:
            break
    if depth % 2 == 0:
        return np.max(minimax_values)
    else:
        return np.min(minimax_values)

def get_leaf_value(node: Node, depth: int):
    opponent_state = node.env.switched_player_state(node.state)
    # the view of the state depends on which level are we on
    if depth % 2 == 0:
        return node.env.get_current_player_score(node.state) - node.env.get_current_player_score(opponent_state)
    else:
        return node.env.get_current_player_score(opponent_state) - node.env.get_current_player_score(node.state) 