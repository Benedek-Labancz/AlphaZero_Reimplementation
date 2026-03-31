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
from AlphaZero.src.training.play import play_episodes, select_az_action

@pytest.mark.parametrize('EnvClass', [TwoDims, ThreeDims, FourDims])
def test_play_episodes(EnvClass):
    env = EnvClass()
    num_squares = env.get_board_size() ** env.get_num_dimensions()
    best_net = PolicyScoreNetwork(num_res_blocks=1, 
                                    input_size=num_squares, 
                                    output_size=num_squares, 
                                    in_channels=6)
    candidate_net = PolicyScoreNetwork(num_res_blocks=1, 
                                    input_size=num_squares, 
                                    output_size=num_squares, 
                                    in_channels=6)
    num_simulations = 20
    num_games = 2
    num_wins = play_episodes(
        env=env,
        best_policy=best_net,
        candidate_policy=candidate_net,
        best_select_action_fn=select_az_action,
        candidate_select_action_fn=select_az_action,
        num_games=num_games,
        num_simulations=num_simulations,
        c=0.5
    )
    
