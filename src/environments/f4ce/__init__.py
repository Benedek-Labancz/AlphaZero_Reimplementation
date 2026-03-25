import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from gymnasium.envs.registration import register
from .two_dims import TwoDims
from .three_dims import ThreeDims
from .four_dims import FourDims


def register_envs():
	register(
		id='4CE-TwoDims',
		entry_point='Self-Play-Lab.src.environments.f4ce.two_dims:TwoDims',
		max_episode_steps=9,
	)

	register(
		id='4CE-ThreeDims',
		entry_point='Self-Play-Lab.src.environments.f4ce.three_dims:ThreeDims',
		max_episode_steps=27,
	)

	register(
		id='4CE-FourDims',
		entry_point='Self-Play-Lab.src.environments.f4ce.four_dims:FourDims',
		max_episode_steps=80,
	)


register_envs()

__all__ = ['TwoDims', 'ThreeDims', 'FourDims']
