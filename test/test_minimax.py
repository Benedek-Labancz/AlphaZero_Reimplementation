import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import numpy as np
from numpy.random import default_rng

from src.mcts.tree import Tree, Node
from src.training.self_play import self_play_episode
from src.environments.f4ce.two_dims import TwoDims
from src.environments.f4ce.three_dims import ThreeDims
from src.environments.f4ce.four_dims import FourDims
from src.policy.network import PolicyScoreNetwork
from AlphaZero.src.training.play import play_episodes, select_az_action
from src.evaluation.minimax import select_minimax_action

GLOBAL_SEED = 42
rng = default_rng(GLOBAL_SEED)

@pytest.mark.parametrize('EnvClass', [TwoDims, ThreeDims, FourDims])
def test_minimax_runs(EnvClass):
    env = EnvClass(render_mode="ansi")
    num_squares = env.get_board_size() ** env.get_num_dimensions()
    candidate_net = PolicyScoreNetwork(num_res_blocks=1, 
                                    input_size=num_squares, 
                                    output_size=num_squares, 
                                    in_channels=6)
    num_simulations = 20
    num_games = 1
    num_wins = play_episodes(
        rng=rng,
        env=env,
        best_policy=None,
        candidate_policy=candidate_net,
        best_select_action_fn=select_minimax_action,
        candidate_select_action_fn=select_az_action,
        num_games=num_games,
        num_simulations=num_simulations,
        c=0.5,
        max_depth=(6 - env.get_num_dimensions())
    )


def _make_state(env, x_plane):
	"""Build a 6-plane binary state with x_plane at X_t (index 0), zeros elsewhere."""
	state = np.zeros([6] + env._num_dimensions * [env._board_size], dtype=int)
	state[env._x_planes[0]] = x_plane
	return state


def test_two_dims_responses():
    env = TwoDims(override=True)

    state = _make_state(env, np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]]))
    action = select_minimax_action(Tree(root=Node(env, state)), max_depth=3)
    assert np.all(action == [0, 2])

    state = _make_state(env, np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]))
    action = select_minimax_action(Tree(root=Node(env, state)), max_depth=3)
    assert np.all(action == [0, 1])

    state = _make_state(env, np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]]))
    action = select_minimax_action(Tree(root=Node(env, state)), max_depth=3)
    assert np.all(action == [0, 2])

    state = _make_state(env, np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]))
    action = select_minimax_action(Tree(root=Node(env, state)), max_depth=6)
    assert np.all(action == [0, 0])