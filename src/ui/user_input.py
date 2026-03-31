import numpy as np

from src.mcts.tree import Tree, Node


def get_user_action(env, state) -> np.array:
	while True:
		inp = input()
		try:
			inp = inp.strip().split(' ')
			coords = list(map(int, inp))
			for num in coords:
				if len(coords) != env.get_num_dimensions():
					raise Exception
				if num <= 0 or num > env.get_board_size():
					raise Exception
			action = np.array([c - 1 for c in coords])
			if not env.is_valid(state, action):
				print(f'Invalid action: {action.tolist()}')
				continue
			break
		except:
			print('Invalid input. Give your move as a list of space-separated integers.')
	return action

def select_user_action(tree: Tree, **kwargs):
	'''
	Prompt the user for an action to take.
	'''
	action = get_user_action(tree.root.env, tree.root.state)
	return action