import torch

from .cnn_tower import CNNTower
from .policy_head import PolicyHead
from .value_head import ValueHead

class PolicyScoreNetwork(torch.nn.Module):
    def __init__(self,
                num_res_blocks: int,
                input_size: int,
                output_size: int,
                in_channels: int,
                hidden_size: int = 256,
                num_filters: list[int] = [256, 2, 1], 
                kernel_size: list[int] = [3, 3, 1], 
                stride: list[int] = [1, 1, 1],
                padding: list[int] = [1, 1, 0],
                *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tower = CNNTower(
            num_res_blocks=num_res_blocks,
            in_channels=in_channels,
            num_filters=num_filters[0],
            kernel_size=kernel_size[0],
            stride=stride[0],
            padding=padding[0]
        )
        self.policy_head = PolicyHead(
            input_size=input_size,
            output_size=output_size,
            in_channels=num_filters[0], # input already has more channels
            num_filters=num_filters[1],
            kernel_size=kernel_size[1],
            stride=stride[1],
            padding=padding[1]
        )
        self.value_head = ValueHead(
            input_size=input_size,
            in_channels=num_filters[0],
            hidden_size=hidden_size,
            num_filters=num_filters[2],
            kernel_size=kernel_size[2],
            stride=stride[2],
            padding=padding[2]
        )

    def forward(self, x: torch.Tensor):
        if x.dim() == 4:
            pass
        elif x.dim() == 3:
            x = self.add_batch_dim(x)
        else:
            raise Exception(f'Input has invalid number of dimensions. 3 or 4 expected, actual: {x.dim()}')
        x = self.tower(x)
        action_values = self.policy_head(x)
        v = self.value_head(x)
        return action_values, v
        
    def add_batch_dim(self, x: torch.Tensor):
        return x.unsqueeze(0)