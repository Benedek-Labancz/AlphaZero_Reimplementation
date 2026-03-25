from typing import Optional
import numpy as np

from .base import BaseEnv
from .render import print_score, print_board


class ThreeDims(BaseEnv):
	_num_dimensions = 3
	num_total_actions = 27

	def __init__(
		self, render_mode: Optional[str] = None, max_timesteps: int = None, **kwargs
	) -> None:
		super().__init__(render_mode, max_timesteps, **kwargs)

	def to_flat_representation(self, state: np.array) -> np.array:
		return state.reshape(self._num_planes, self._board_size, -1)

	def _get_scoring_cases(self) -> np.array:
		"""
		Compute coordinate triplets.
		Resulting array will be of shape (N, 3, 3)
		"""

		u = np.array(self._board_size * [1]).reshape(self._board_size, 1)
		roll = np.arange(self._board_size).reshape(self._board_size, 1)

		# 27 cases
		columns = np.concatenate(
			[
				np.array(
					[
						np.concatenate((roll, i * u, j * u), axis=1)
						for i in range(self._board_size)
						for j in range(self._board_size)
					]
				),
				np.array(
					[
						np.concatenate((i * u, roll, j * u), axis=1)
						for i in range(self._board_size)
						for j in range(self._board_size)
					]
				),
				np.array(
					[
						np.concatenate((i * u, j * u, roll), axis=1)
						for i in range(self._board_size)
						for j in range(self._board_size)
					]
				),
			]
		)

		diagonals = np.concatenate(
			[
				# 18 cases (3 x 3 x 2)
				np.array(
					[np.concatenate((i * u, roll, roll), axis=1) for i in range(self._board_size)]
				),
				np.array(
					[
						np.concatenate((i * u, roll, np.flip(roll, axis=0)), axis=1)
						for i in range(self._board_size)
					]
				),
				np.array(
					[np.concatenate((roll, i * u, roll), axis=1) for i in range(self._board_size)]
				),
				np.array(
					[
						np.concatenate((roll, i * u, np.flip(roll, axis=0)), axis=1)
						for i in range(self._board_size)
					]
				),
				np.array(
					[np.concatenate((roll, roll, i * u), axis=1) for i in range(self._board_size)]
				),
				np.array(
					[
						np.concatenate((roll, np.flip(roll, axis=0), i * u), axis=1)
						for i in range(self._board_size)
					]
				),
				# 4 cases
				np.array([np.concatenate((roll, roll, roll), axis=1)]),
				np.array([np.concatenate((roll, roll, np.flip(roll, axis=0)), axis=1)]),
				np.array([np.concatenate((roll, np.flip(roll, axis=0), roll), axis=1)]),
				np.array(
					[
						np.concatenate(
							(
								roll,
								np.flip(roll, axis=0),
								np.flip(roll, axis=0),
							),
							axis=1,
						)
					]
				),
			]
		)

		return np.concatenate((columns, diagonals), axis=0)

	def render(self):
		if self._render_mode in ['ansi']:
			print_score(self._score)
			for i in range(self._board_state.shape[0]):
				print()
				print_board(self._board_state[i])
