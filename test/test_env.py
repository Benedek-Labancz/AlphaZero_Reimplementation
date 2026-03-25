import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import numpy as np
from src.environments.f4ce.two_dims import TwoDims
from src.environments.f4ce.three_dims import ThreeDims
from src.environments.f4ce.four_dims import FourDims


env2 = TwoDims()
env3 = ThreeDims()
env4 = FourDims()


@pytest.mark.parametrize('EnvClass', [TwoDims, ThreeDims, FourDims])
def test_step_changes_board_and_switches_player(EnvClass):
	env = EnvClass()
	board_before = env.get_board_state().copy()
	action = np.zeros(env.get_num_dimensions(), dtype=int)
	env.step(action)

	board_state = env.get_board_state()
	assert not np.array_equal(board_state, board_before)
	assert np.array_equal(board_state[env._x_planes, ...], board_before[env._y_planes, ...])
	assert np.array_equal(board_state[env._y_planes[1], ...], board_before[env._x_planes[0], ...])
	assert np.array_equal(board_state[env._i_plane, ...], board_before[env._i_plane, ...])
	assert not np.array_equal(board_state[env._c_plane, ...], board_before[env._c_plane, ...])


@pytest.mark.parametrize('EnvClass', [TwoDims, ThreeDims, FourDims])
def test_simulate_step_does_not_mutate_state(EnvClass):
	env = EnvClass()
	board_before = env.get_board_state().copy()
	action = np.zeros(env.get_num_dimensions(), dtype=int)

	env.simulate_step(env.get_board_state(), action)

	board_state = env.get_board_state()
	assert np.array_equal(board_state, board_before)


@pytest.mark.parametrize('EnvClass', [TwoDims, ThreeDims, FourDims])
def test_full_board_is_terminal(EnvClass):
	env = EnvClass()
	invalid = env.get_board_state()[env._i_plane, ...]
	current = np.random.randint(2, size=env.get_num_dimensions() * (env.get_board_size(),))
	current = current * (1 - invalid)
	opponent = (1 - current) * (1 - invalid)
	env._board_state[env._x_planes[0], ...] = current.copy()
	env._board_state[env._y_planes[0], ...] = opponent.copy()
	assert env.is_terminal(env.get_board_state())


@pytest.mark.parametrize('EnvClass', [TwoDims, ThreeDims, FourDims])
def test_invalid_action_occupied_square(EnvClass):
	env = EnvClass()
	action = np.zeros(env.get_num_dimensions(), dtype=int)
	env.step(action)
	with pytest.raises(Exception):
		env.step(action)
		env.step(action)


@pytest.mark.parametrize('EnvClass', [TwoDims, ThreeDims, FourDims])
def test_invalid_action_out_of_bounds(EnvClass):
	env = EnvClass()
	action = env._board_size * np.ones(env.get_num_dimensions(), dtype=int)
	with pytest.raises((Exception, AssertionError)):
		env.step(action)


@pytest.mark.parametrize('EnvClass', [TwoDims, ThreeDims, FourDims])
def test_initial_score_is_zero(EnvClass):
	env = EnvClass()
	state = env._board_state
	assert env.get_current_player_score(state) == 0
	assert env.get_current_player_score(env.switched_player_state(state)) == 0


@pytest.mark.parametrize('EnvClass', [TwoDims, ThreeDims, FourDims])
def test_step_state_representation(EnvClass):
	env = EnvClass()
	old_state = env._board_state.copy()
	action = np.zeros(env.get_num_dimensions(), dtype=int)

	obs, _, _, _, _ = env.step(action)

	# _x_planes of obs equals old _y_planes
	assert np.array_equal(obs[env._x_planes], old_state[env._y_planes])

	# _y_planes[0] of obs differs from old _x_planes[0] at exactly one place (the action)
	diff = obs[env._y_planes[0], ...] != old_state[env._x_planes[0], ...]
	assert diff.sum() == 1

	# _y_planes[1] of obs equals old _x_planes[0]
	assert np.array_equal(obs[env._y_planes[1]], old_state[env._x_planes[0]])

	# _i_plane unchanged
	assert np.array_equal(obs[env._i_plane], old_state[env._i_plane])

	# _c_plane changed
	assert not np.array_equal(obs[env._c_plane], old_state[env._c_plane])


def _make_state(env, x_plane):
	"""Build a 6-plane binary state with x_plane at X_t (index 0), zeros elsewhere."""
	state = np.zeros([6] + env._num_dimensions * [env._board_size], dtype=int)
	state[env._x_planes[0]] = x_plane
	return state


def test_two_dims_scoring():
	assert (
		env2.get_current_player_score(
			_make_state(env2, np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]]))
		)
		== 1
	)
	assert (
		env2.get_current_player_score(
			_make_state(env2, np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]]))
		)
		== 1
	)
	assert (
		env2.get_current_player_score(
			_make_state(env2, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
		)
		== 1
	)
	assert (
		env2.get_current_player_score(
			_make_state(env2, np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]))
		)
		== 1
	)
	assert (
		env2.get_current_player_score(
			_make_state(env2, np.array([[1, 1, 1], [0, 1, 0], [0, 0, 1]]))
		)
		== 2
	)
	assert (
		env2.get_current_player_score(
			_make_state(env2, np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]]))
		)
		== 3
	)
	# q pieces at (0,0),(1,0),(1,1),(2,2); p pieces at (0,1),(0,2),(1,2)
	assert (
		env2.get_current_player_score(
			_make_state(env2, np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]]))
		)
		== 0
	)
	assert (
		env2.get_current_player_score(
			_make_state(env2, np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]]))
		)
		== 1
	)


def test_three_dims_scoring():
	assert (
		env3.get_current_player_score(
			_make_state(
				env3,
				np.array(
					[
						[[1, 0, 0], [1, 0, 0], [1, 0, 0]],
						[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
						[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
					]
				),
			)
		)
		== 1
	)
	assert (
		env3.get_current_player_score(
			_make_state(
				env3,
				np.array(
					[
						[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
						[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
						[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
					]
				),
			)
		)
		== 1
	)
	assert (
		env3.get_current_player_score(
			_make_state(
				env3,
				np.array(
					[
						[[1, 1, 1], [0, 0, 0], [0, 0, 0]],
						[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
						[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
					]
				),
			)
		)
		== 1
	)
	assert (
		env3.get_current_player_score(
			_make_state(
				env3,
				np.array(
					[
						[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
						[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
						[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
					]
				),
			)
		)
		== 1
	)
	assert (
		env3.get_current_player_score(
			_make_state(
				env3,
				np.array(
					[
						[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
						[[0, 0, 0], [1, 0, 0], [0, 0, 0]],
						[[0, 0, 0], [0, 0, 0], [1, 0, 0]],
					]
				),
			)
		)
		== 1
	)
	assert (
		env3.get_current_player_score(
			_make_state(
				env3,
				np.array(
					[
						[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
						[[0, 1, 0], [0, 0, 0], [0, 0, 0]],
						[[0, 0, 1], [0, 0, 0], [0, 0, 0]],
					]
				),
			)
		)
		== 1
	)
	assert (
		env3.get_current_player_score(
			_make_state(
				env3,
				np.array(
					[
						[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
						[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
						[[0, 0, 0], [0, 0, 0], [0, 0, 1]],
					]
				),
			)
		)
		== 1
	)
	assert env3.get_current_player_score(_make_state(env3, np.ones((3, 3, 3), dtype=int))) == 49
	assert (
		env3.get_current_player_score(
			_make_state(
				env3,
				np.array(
					[
						[[0, 1, 1], [1, 0, 1], [1, 1, 0]],
						[[1, 0, 1], [0, 0, 0], [1, 0, 1]],
						[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
					]
				),
			)
		)
		== 0
	)


@pytest.mark.parametrize('EnvClass', [TwoDims, ThreeDims, FourDims])
def test_action_mask_excludes_opponent_and_invalid(EnvClass):
	env = EnvClass()
	state = env._board_state.copy()
	board_shape = env._num_dimensions * [env._board_size]

	# Place current player's piece at index (0, 0, ...)
	current_idx = tuple([0] * env._num_dimensions)
	state[env._x_planes[0]][current_idx] = 1

	# Place opponent's piece at index (1, 0, ...)
	opponent_idx = tuple([1] + [0] * (env._num_dimensions - 1))
	state[env._y_planes[0]][opponent_idx] = 1

	# Mark a square as invalid at index (2, 0, ...)
	invalid_idx = tuple([2] + [0] * (env._num_dimensions - 1))
	state[env._i_plane][invalid_idx] = 1

	mask = env.get_action_mask(state)

	assert mask[current_idx] == False, "Current player's square should be masked out"
	assert mask[opponent_idx] == False, "Opponent's square should be masked out"
	assert mask[invalid_idx] == False, 'Invalid square should be masked out'

	# A square untouched by any of the above should be available
	free_idx = tuple([env._board_size - 1] * env._num_dimensions)
	assert mask[free_idx] == True, 'Unoccupied valid square should be available'


def test_four_dims_scoring():
	assert env4.get_current_player_score(_make_state(env4, np.zeros((3, 3, 3, 3), dtype=int))) == 0
	assert (
		env4.get_current_player_score(
			_make_state(
				env4,
				np.array(
					[
						[
							[[1, 0, 0], [1, 0, 0], [1, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
						],
						[
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
						],
						[
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
						],
					]
				),
			)
		)
		== 1
	)
	assert (
		env4.get_current_player_score(
			_make_state(
				env4,
				np.array(
					[
						[
							[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 1]],
						],
						[
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
						],
						[
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
						],
					]
				),
			)
		)
		== 1
	)
	assert (
		env4.get_current_player_score(
			_make_state(
				env4,
				np.array(
					[
						[
							[[0, 0, 0], [1, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
						],
						[
							[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
						],
						[
							[[0, 0, 0], [0, 0, 1], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
						],
					]
				),
			)
		)
		== 1
	)
	assert (
		env4.get_current_player_score(
			_make_state(
				env4,
				np.array(
					[
						[
							[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
						],
						[
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
							[[0, 1, 0], [0, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
						],
						[
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
							[[0, 0, 1], [0, 0, 0], [0, 0, 0]],
						],
					]
				),
			)
		)
		== 1
	)
	assert (
		env4.get_current_player_score(
			_make_state(
				env4,
				np.array(
					[
						[
							[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
						],
						[
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
						],
						[
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
						],
					]
				),
			)
		)
		== 8
	)
	assert (
		env4.get_current_player_score(
			_make_state(
				env4,
				np.array(
					[
						[
							[[1, 0, 1], [0, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
						],
						[
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
						],
						[
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
						],
					]
				),
			)
		)
		== 0
	)
	assert (
		env4.get_current_player_score(
			_make_state(
				env4,
				np.array(
					[
						[
							[[0, 0, 0], [1, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
						],
						[
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [1, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
						],
						[
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [1, 0, 0], [0, 0, 0]],
							[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
						],
					]
				),
			)
		)
		== 0
	)
