import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import numpy as np

from src.mcts.tree import Tree, Node
from src.mcts.search import run_simulation, run_mcts
from src.environments.f4ce.two_dims import TwoDims
from src.environments.f4ce.three_dims import ThreeDims
from src.environments.f4ce.four_dims import FourDims
from src.policy.network import PolicyScoreNetwork



@pytest.mark.parametrize('EnvClass', [TwoDims, ThreeDims, FourDims])
def test_simulation(EnvClass):
    env = EnvClass()
    root_node = Node(env=env, state=env.get_board_state())
    tree = Tree(root=root_node)
    num_squares = env.get_board_size() ** env.get_num_dimensions()
    policy_net = PolicyScoreNetwork(num_res_blocks=1, 
                                    input_size=num_squares, 
                                    output_size=num_squares, 
                                    in_channels=6)
    new_tree = run_simulation(
        tree=tree,
        policy=policy_net,
        c=0.5
    )



@pytest.mark.parametrize('EnvClass', [TwoDims, ThreeDims, FourDims])
def test_mcts(EnvClass):
    env = EnvClass()
    root_node = Node(env=env, state=env.get_board_state())
    tree = Tree(root=root_node)
    num_squares = env.get_board_size() ** env.get_num_dimensions()
    policy_net = PolicyScoreNetwork(num_res_blocks=1, 
                                    input_size=num_squares, 
                                    output_size=num_squares, 
                                    in_channels=6)
    num_simulations = 200
    tree, pi = run_mcts(
        tree=tree,
        policy=policy_net,
        num_simulations=num_simulations,
        c=0.5,
        tau=0.1
    )
    assert len(pi.reshape(-1)) == len(root_node.valid_actions)
    # assert len(tree.nodes) == num_simulations