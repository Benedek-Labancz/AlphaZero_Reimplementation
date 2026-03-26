import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import numpy as np

from src.mcts.tree import Tree, Node
from src.training.self_play import play_episode
from src.environments.f4ce.two_dims import TwoDims
from src.environments.f4ce.three_dims import ThreeDims
from src.environments.f4ce.four_dims import FourDims
from src.policy.network import PolicyScoreNetwork
from src.training.evaluate import evaluate_policy

@pytest.mark.parametrize('EnvClass', [TwoDims, ThreeDims, FourDims])
def test_evaluate_policy(EnvClass):
    env = EnvClass()
    num_squares = env.get_board_size() ** env.get_num_dimensions()
    best_net = PolicyScoreNetwork(num_res_blocks=4, 
                                    input_size=num_squares, 
                                    output_size=num_squares, 
                                    in_channels=6)
    candidate_net = PolicyScoreNetwork(num_res_blocks=4, 
                                    input_size=num_squares, 
                                    output_size=num_squares, 
                                    in_channels=6)
    num_simulations = 20
    num_games = 10
    win_rate = evaluate_policy(
        env=env,
        best_policy=best_net,
        candidate_policy=candidate_net,
        num_games=num_games,
        num_simulations=num_simulations,
        c=0.5
    )
    
