import torch
from torch.nn import Sequential, Conv2d, ReLU, BatchNorm2d
from copy import deepcopy

class CNNTower(torch.nn.Module):
    def __init__(self,
                 num_res_blocks: int,
                 in_channels: int, 
                 num_filters: int = 256, 
                 kernel_size: int = 3, 
                 stride: int = 1,
                 padding: int = 1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv_block = Sequential(
            Conv2d(in_channels=in_channels, out_channels=num_filters, kernel_size=kernel_size, stride=stride, padding=padding),
            BatchNorm2d(num_features=num_filters),
            ReLU()
        )
        # This block contains the operations before the skip connection
        conv_parameters = {"in_channels":num_filters, "out_channels":num_filters, "kernel_size":kernel_size, "stride":stride, "padding":padding}
        res_block = Sequential(
            Conv2d(**conv_parameters),
            BatchNorm2d(num_features=num_filters),
            ReLU(),
            Conv2d(**conv_parameters),
            BatchNorm2d(num_features=num_filters)
        )
        self.res_blocks = []
        for i in range(num_res_blocks):
            block = deepcopy(res_block)
            self.add_module(f'res_block_{i + 1}', block)
            self.res_blocks.append(block)
    
    def forward(self, x: torch.Tensor):
        x = self.conv_block(x)
        for res_block in self.res_blocks:
            x = x + res_block(x)
            x = torch.nn.functional.relu(x)
        return x