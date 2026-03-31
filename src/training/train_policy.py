import torch
from torch.nn.functional import mse_loss, cross_entropy
import numpy as np

def compute_loss(predicted_logits: torch.Tensor, v: torch.Tensor, pi_values: torch.Tensor, z: torch.Tensor):
    return mse_loss(v, z) + cross_entropy(predicted_logits, pi_values) # L2 regularization added later by optimizer

def train_batch(data, policy, optimizer, scheduler):
    """
    Mini-batch training step,
    returning the average loss on the mini-batch.
    """
    states, pi_values, winners = data
    states = torch.as_tensor(states, dtype=torch.float32)
    pi_values = torch.as_tensor(pi_values, dtype=torch.float32)
    winners = torch.as_tensor(winners, dtype=torch.float32)
    optimizer.zero_grad()
    p, v = policy(states)
    loss = compute_loss(
        predicted_logits=p,
        v=v.squeeze(),
        pi_values=pi_values,
        z=winners
    )
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.mean()
