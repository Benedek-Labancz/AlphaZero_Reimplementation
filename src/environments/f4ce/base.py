from abc import ABC, abstractmethod
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional


class BaseEnv(gym.Env, ABC):
	_metadata = {'render_modes': ['human', 'ansi']}
	_board_size = 3
	_num_dimensions = None

	_num_planes = 6

	# Plane indices of the board representation
	_x_planes = [0, 1]
	_y_planes = [2, 3]
	_i_plane = 4
	_c_plane = 5

	def __init__(
		self,
		render_mode: Optional[str] = None,
		max_timesteps: Optional[int] = None,
		**kwargs,
	) -> None:
		super().__init__()

		self._config = None

		self._render_mode = render_mode
		self._max_timesteps = np.inf if max_timesteps is None else max_timesteps

		# An observation consists of 6 binary masks:
		# (X_t, X_t-1, Y_t, Y_t-1, I, C)
		self.observation_space = spaces.MultiDiscrete(
			2 * np.ones([self._num_planes] + (self._num_dimensions * [self._board_size]))
		)

		# Actions are represented by coordinates
		self.action_space = spaces.MultiDiscrete(self._num_dimensions * [self._board_size])

		# To be defined by subclass
		self._initial_state = self._get_initial_board()
		self._board_state = self._initial_state.copy()
		self._scoring_cases = self._get_scoring_cases()

		self._last_action_to_render = None

		# Reset the env to reset timesteps, initialize scores,
		# current and next player
		self.reset()

	def set_config(self, config: dict) -> None:
		self._config = config

	def get_metadata(cls) -> None:
		return cls._metadata

	def get_board_size(cls) -> None:
		return cls._board_size

	def get_num_dimensions(cls) -> None:
		return cls._num_dimensions

	def get_board_state(self) -> np.array:
		return self._board_state

	def get_action_mask(self, state: np.array) -> np.array:
		"""
		Return a mask of shape num_dimensions * [board_size]
		"""
		# TODO: invalidate mirror moves, perhaps as defined by subclasses
		return state[[self._x_planes[0], self._y_planes[0], self._i_plane], ...].sum(axis=0) == 0

	def _get_reward(self, state: np.array, action: np.array) -> float:
		return 0

	def _get_info(self) -> dict:
		return {}

	def reset(
		self, seed: Optional[int] = None, options: Optional[dict] = None
	) -> tuple[dict, dict]:
		"""
		Sets the random seed, and resets the timestep,
		resets the board to its initial position,
		zeros out the scores and resets the current player.
		"""
		super().reset(seed=seed)

		self._board_state = self._initial_state.copy()
		info = self._get_info()

		return self._board_state, info

	def switched_player_state(self, state: np.array) -> np.array:
		"""
		Permute feature planes and change color plane in state.
		"""
		new_state = state[self._y_planes + self._x_planes + [self._i_plane, self._c_plane], ...]
		new_state[self._c_plane] = 1 - new_state[self._c_plane]
		return new_state

	def simulate_step(
		self, state: np.array, action: np.array
	) -> tuple[np.array, float] | Exception:
		"""
		Main environment logic.

		Checks if action is valid, executes the action,
		determines the reward,
		asserts terminal state and truncation, switches players.
		"""
		if not self.is_valid(action):
			raise Exception(f'Invalid action {tuple(action)} encountered.')

		# Make sure that passed-in state is not overwritten
		state_copy = state.copy()
		# Move X_t plane in place of the X_t-1 plane ("as time passes")
		state_copy[self._x_planes[1], ...] = state_copy[self._x_planes[0], ...].copy()
		# Flip bit to play move
		state_copy[self._x_planes[0], *action] = 1
		observation = self.switched_player_state(state_copy)
		reward = self._get_reward(state, action)
		terminated = self.is_terminal(state_copy)
		truncated = (
			state_copy[[self._x_planes[0], self._y_planes[0]], ...].sum() > self._max_timesteps
		)
		info = self._get_info()
		return observation, reward, terminated, truncated, info

	def step(self, action: np.array) -> tuple[dict, float, bool, bool, dict] | Exception:
		"""
		Uses environment dynamics to simulate a step and executes that step.
		"""
		if not self.is_valid(action):
			raise Exception(f'Invalid action {tuple(action)} encountered.')
		observation, reward, terminated, truncated, info = self.simulate_step(
			self._board_state, action
		)
		self._board_state = observation.copy()
		if self._render_mode is not None:
			self._last_action_to_render = action.copy()
		return observation, reward, terminated, truncated, info

	def is_valid(self, action: np.array) -> bool:
		try:
			if action.size != self._num_dimensions:
				return False
			return self.get_action_mask(self._board_state)[*action] == 1
		except AttributeError as e:
			raise Exception(f'Action {action} could not be validated; {e}.')

	def is_terminal(self, state: np.array) -> bool:
		return state[[self._x_planes[0], self._y_planes[0], self._i_plane], ...].sum() == (
			self._board_size**self._num_dimensions
		)

	def get_winner(self, state: np.array = None) -> int | None:
		"""
		-1 if current player lost,
		 1 if current player won,
		 0 otherwise
		"""
		current_player_score = self.get_current_player_score(state)
		opponent_score = self.get_current_player_score(self.switched_player_state(state))
		if opponent_score > current_player_score:
			return -1
		elif current_player_score > opponent_score:
			return 1
		else:
			return 0

	def get_current_player_score(self, state: np.array) -> float:
		# Bring the last axis to the front. This is where we can index into the array
		scoring_positions = np.transpose(
			self._scoring_cases, axes=(2, 0, 1)
		)  # (N, board_size, num_dimensions) -> (um_dimensions, N, board_size)
		is_at_position = state[self._x_planes[0], *scoring_positions]  # (N, 3)
		scores = np.all(is_at_position, axis=1).astype(int)  # (N)
		total_score = scores.sum()
		return total_score

	def _get_initial_board(self) -> np.array:
		return np.zeros([self._num_planes] + (self._num_dimensions * [self._board_size]))

	@abstractmethod
	def to_flat_representation(self, state: np.array) -> np.array:
		"""
		Convert the board representation to have 2 board dimensions
		so that it can be handled as an image.
		"""
		return state

	@abstractmethod
	def _get_scoring_cases(self) -> np.array:
		return np.array()

	@abstractmethod
	def render(self):
		return
