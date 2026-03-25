import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pytest

from src.policy.network import PolicyScoreNetwork

net = PolicyScoreNetwork(num_res_blocks=4, input_size=81, output_size=81, in_channels=6)

def test_with_unbatched_input():
    dummy = torch.rand(6, 9, 9)
    probs, v = net(dummy)
    assert probs.shape == (1, 81)
    assert v.shape == (1, 1)

def test_with_batched_input():
    dummy = torch.rand(64, 6, 9, 9)
    probs, v = net(dummy)
    assert probs.shape == (64, 81)
    assert v.shape == (64, 1)