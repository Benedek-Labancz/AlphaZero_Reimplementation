"""
Utilities for pretty printing.
"""

import re
import numpy as np

class RoleEnum:
	X = 0
	O = 1

class BoardEnum:
	X = "X"
	O = "O"

_ANSI_RE = re.compile(r'\033\[[0-9;]*m')

# Letter pixel masks — █ = letter stroke (rendered as space), else background (rendered as /)
_L4 = [
	'  █   ',
	' ██   ',
	'█ █   ',
	'██████',
	'  █   ',
	'  █   ',
]
_LC = [
	' █████',
	'██    ',
	'█     ',
	'█     ',
	'██    ',
	' █████',
]
_LE = [
	'██████',
	'█     ',
	'█████ ',
	'█     ',
	'█     ',
	'██████',
]


def print_banner():
	W = 76  # content width
	SLANT = 1  # italic shift per row (lean right going up)
	GAP = 5  # space between letters
	LETTER_W = 6
	letters = [_L4, _LC, _LE]
	H = len(_L4)

	total_w = len(letters) * LETTER_W + (len(letters) - 1) * GAP
	base_x = (W - total_w) // 2  # anchor at the bottom row
	starts = [base_x + i * (LETTER_W + GAP) for i in range(len(letters))]

	rows = []
	rows.append('/' * W)  # top padding row
	for r in range(H):
		shift = (H - 1 - r) * SLANT
		row = list('/' * W)
		for li, letter in enumerate(letters):
			lx = starts[li] + shift
			for cx in range(LETTER_W):
				if 0 <= lx + cx < W and letter[r][cx] == '█':
					row[lx + cx] = ' '
		rows.append(''.join(row))
	rows.append('/' * W)  # bottom padding row

	top = '┌' + '─' * (W + 2) + '┐'
	bot = '└' + '─' * (W + 2) + '┘'
	print(top)
	for row in rows:
		print('│ ' + row + ' │')
	print(bot)
	print()


def _vlen(s: str) -> int:
	"""Visible length of a string, ignoring ANSI escape codes."""
	return len(_ANSI_RE.sub('', s))


def _vcenter(s: str, width: int) -> str:
	"""Center s in a field of `width` visible characters."""
	pad = width - _vlen(s)
	left_pad = pad // 2
	right_pad = pad - left_pad
	return ' ' * left_pad + s + ' ' * right_pad


def _box(lines: list[str]) -> list[str]:
	w = max(_vlen(line) for line in lines)
	top = '┌' + '─' * (w + 2) + '┐'
	bottom = '└' + '─' * (w + 2) + '┘'
	mid = ['│ ' + _vcenter(line, w) + ' │' for line in lines]
	return [top] + mid + [bottom]


def print_line(text: str):
	for line in _box([text]):
		print(line)
	print()


def print_score(score: dict, opponent_name: str = 'Computer'):
	x_score = score[RoleEnum.X]
	o_score = score[RoleEnum.O]

	left = _box(['\033[1;31mYOU\033[0m', _X])
	center = _box([f'\033[1m{x_score}\033[0m  —  \033[1m{o_score}\033[0m'])
	right = _box([f'\033[1m{opponent_name}\033[0m', _O])

	# pad all panels to the same height
	h = max(len(left), len(center), len(right))
	for panel in (left, center, right):
		inner_w = _vlen(panel[0]) - 2  # exclude the two border chars
		while len(panel) < h:
			panel.insert(-1, '│' + ' ' * inner_w + '│')

	gap = '   '
	for parts in zip(left, center, right):
		print(gap.join(parts))
	print()


def print_board(board: np.array):
	colored = [['', '', ''], ['', '', ''], ['', '', '']]
	for i in range(board.shape[0]):
		for j in range(board.shape[1]):
			# TODO: add actual coloring
			# TODO: solve mapping
			if board[i][j] == BoardEnum.X:
				colored[i][j] = 'X'
			elif board[i][j] == BoardEnum.O:
				colored[i][j] = 'O'
			else:
				colored[i][j] = '-'

	print('     ' + colored[0][0] + ' │ ' + colored[0][1] + ' │ ' + colored[0][2])
	print('    ───┼───┼───')
	print('     ' + colored[1][0] + ' │ ' + colored[1][1] + ' │ ' + colored[1][2])
	print('    ───┼───┼───')
	print('     ' + colored[2][0] + ' │ ' + colored[2][1] + ' │ ' + colored[2][2])


_X = '\033[1;31mX\033[0m'  # bold red
_O = '\033[1;34mO\033[0m'  # bold blue
_INV = '\033[2m~\033[0m'  # dim
_HL = '\033[43m'  # yellow background
_RST = '\033[0m'


def _cell(val, highlight: bool = False) -> str:
	if val == BoardEnum.X:
		symbol = '\033[1;31mX\033[0m'
	elif val == BoardEnum.O:
		symbol = '\033[1;34mO\033[0m'
	elif val == BoardEnum.INVALID:
		symbol = '\033[2m~\033[0m'
	else:
		symbol = '-'
	if highlight:
		return _HL + symbol + _RST
	return symbol


def _render_mini_board(
	board: np.array,
	macro_row: int,
	macro_col: int,
	last_action: np.array = None,
) -> list[str]:
	"""Return the lines that make up one 3x3 sub-board (border included)."""
	sub = board[macro_row][macro_col]
	# Determine which micro cell to highlight (if the last action is in this sub-board)
	hl_row, hl_col = -1, -1
	if last_action is not None and last_action[0] == macro_row and last_action[1] == macro_col:
		hl_row, hl_col = last_action[2], last_action[3]

	rows = []
	for micro_row in range(3):
		c = sub[micro_row]
		cells = [
			_cell(c[col], highlight=(micro_row == hl_row and col == hl_col)) for col in range(3)
		]
		rows.append(f' {cells[0]} │ {cells[1]} │ {cells[2]} ')
		if micro_row < 2:
			rows.append('───┼───┼───')
	inner_w = 11  # visible chars per content line
	top = '┌' + '─' * inner_w + '┐'
	bottom = '└' + '─' * inner_w + '┘'
	return [top] + ['│' + r + '│' for r in rows] + [bottom]


def print_board_3x3x3x3(board: np.array, last_action: np.array = None):
	# board shape: (3, 3, 3, 3) — [macro_row, macro_col, micro_row, micro_col]
	gap = '   '  # horizontal gap between sub-boards

	for macro_row in range(3):
		if macro_row > 0:
			print()  # vertical gap between macro rows
		panels = [_render_mini_board(board, macro_row, mc, last_action) for mc in range(3)]
		for line_idx in range(len(panels[0])):
			print(gap.join(p[line_idx] for p in panels))
