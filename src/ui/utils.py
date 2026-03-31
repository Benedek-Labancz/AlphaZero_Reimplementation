import os

def clear_terminal() -> None:
	if os.name == 'nt':
		os.system('cls')
	else:
		os.system('clear')
