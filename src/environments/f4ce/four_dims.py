import numpy as np
from typing import Optional
from .base import BaseEnv


class FourDims(BaseEnv):
	_num_dimensions = 4
	num_total_actions = 80

	def __init__(
		self, render_mode: Optional[str] = None, max_timesteps: int = None, **kwargs
	) -> None:
		super().__init__(render_mode, max_timesteps, **kwargs)

	def to_flat_representation(self, state: np.array) -> np.array:
		return state.reshape(self._num_planes, self._board_size**2, self._board_size**2)

	def _get_initial_board(self):
		board = np.zeros([6] + (self._num_dimensions * [self._board_size]))
		# The very middle cell is invalidated by the game rules
		middle_idx = self._board_size // 2
		board[self._i_plane, *([middle_idx] * self._num_dimensions)] = 1
		return board

	def _get_scoring_cases(self) -> np.array:
		"""
		Compute scoring cases, shape (N, 3, 4)
		"""
		u = np.array(self._board_size * [1]).reshape(self._board_size, 1)
		roll = np.arange(self._board_size).reshape(self._board_size, 1)

		two_horizontals = np.array(
			[np.concatenate((i * u, roll), axis=1) for i in range(self._board_size)]
		)

		two_verticals = np.array(
			[np.concatenate((roll, i * u), axis=1) for i in range(self._board_size)]
		)

		two_diagonals = np.array(
			[
				np.concatenate((roll, roll), axis=1),
				np.concatenate((roll, np.flip(roll, axis=0)), axis=1),
			]
		)

		two_coords = np.concatenate((two_horizontals, two_verticals, two_diagonals), axis=0)

		finals = []

		# Enumerating the cases when all three are in the same 3x3
		for i in range(self._board_size):
			for j in range(self._board_size):
				sq_ind = np.array([i, j] * (len(two_coords) * self._board_size)).reshape(
					len(two_coords), self._board_size, 2
				)
				finals.append(np.concatenate((sq_ind, two_coords), axis=2))

		# Enumerating the cases when the three are spread across a cube
		for k in range(len(two_coords)):
			active = two_coords[k]
			final_columns = np.concatenate(
				[
					np.array(
						[
							np.concatenate((active, i * u, j * u), axis=1)
							for i in range(self._board_size)
							for j in range(self._board_size)
						]
					),
					np.array(
						[
							np.concatenate((active, roll, i * u), axis=1)
							for i in range(self._board_size)
						]
					),
					np.array(
						[
							np.concatenate((active, i * u, roll), axis=1)
							for i in range(self._board_size)
						]
					),
				]
			)

			final_diagonals = np.concatenate(
				[
					np.array([np.concatenate((active, roll, roll), axis=1)]),
					np.array([np.concatenate((active, roll, np.flip(roll, axis=0)), axis=1)]),
					np.array([np.concatenate((active, np.flip(roll, axis=0), roll), axis=1)]),
					np.array(
						[
							np.concatenate(
								(
									active,
									np.flip(roll, axis=0),
									np.flip(roll, axis=0),
								),
								axis=1,
							)
						]
					),
					np.array(
						[
							np.concatenate((active, i * u, np.flip(roll, axis=0)), axis=1)
							for i in range(self._board_size)
						]
					),
					np.array(
						[
							np.concatenate((active, np.flip(roll, axis=0), i * u), axis=1)
							for i in range(self._board_size)
						]
					),
				]
			)
			finals.append(final_columns)
			finals.append(final_diagonals)
		return np.concatenate(finals, axis=0)

	def render(self):
		pass