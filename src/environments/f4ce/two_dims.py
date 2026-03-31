import numpy as np
from typing import Optional
from .base import BaseEnv

from src.ui.utils import clear_terminal

class TwoDims(BaseEnv):
	_num_dimensions = 2
	num_total_actions = 9

	def __init__(
		self, render_mode: Optional[str] = None, max_timesteps: int = None, **kwargs
	) -> None:
		super().__init__(render_mode, max_timesteps, **kwargs)

	def to_flat_representation(self, state: np.array) -> np.array:
		return state

	def _get_scoring_cases(self) -> np.array:
		"""
		Compute all the N cases of coordinate triplets
		that score a point. The resulting array will be
		of shape (N, 3, 2).
		"""

		u_base = np.array(self._board_size * [1]).reshape(self._board_size, 1)
		roll_base = np.arange(self._board_size).reshape(self._board_size, 1)

		horizontals = np.array(
			[np.concatenate((i * u_base, roll_base), axis=1) for i in range(self._board_size)]
		)

		verticals = np.array(
			[np.concatenate((roll_base, i * u_base), axis=1) for i in range(self._board_size)]
		)

		diagonals = np.array(
			[
				np.concatenate((roll_base, roll_base), axis=1),
				np.concatenate((roll_base, np.flip(roll_base, axis=0)), axis=1),
			]
		)

		return np.concatenate((horizontals, verticals, diagonals), axis=0)

	def render(self):
		if self.render_mode == 'ansi':
			clear_terminal()
			print(self._board_state)
