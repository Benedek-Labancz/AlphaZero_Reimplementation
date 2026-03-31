import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import numpy as np

from src.mcts.tree import Tree, Node
from src.training.self_play import self_play_episode
from src.environments.f4ce.two_dims import TwoDims
from src.environments.f4ce.three_dims import ThreeDims
from src.environments.f4ce.four_dims import FourDims
from src.policy.network import PolicyScoreNetwork

@pytest.mark.parametrize('EnvClass', [TwoDims, ThreeDims, FourDims])
def test_play_episode(EnvClass):
    env = EnvClass()
    root_node = Node(env=env, state=env.get_board_state())
    tree = Tree(root=root_node)
    num_squares = env.get_board_size() ** env.get_num_dimensions()
    policy_net = PolicyScoreNetwork(num_res_blocks=4, 
                                    input_size=num_squares, 
                                    output_size=num_squares, 
                                    in_channels=6)
    num_simulations = 20
    ep_states, ep_pi_values, ep_winners = self_play_episode(
        tree=tree,
        policy=policy_net,
        num_simulations=num_simulations,
        c=0.5,
        tau=1,
        early_selection_threshold=30
    )

    assert len(ep_states) == len(ep_pi_values)
    assert len(ep_pi_values) == len(ep_winners)