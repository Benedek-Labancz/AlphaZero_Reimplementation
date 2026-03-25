import torch
from torch.nn import Sequential, Linear, Conv2d, BatchNorm2d, ReLU

class PolicyHead(torch.nn.Module):
    def __init__(self, 
                 input_size: int,
                 output_size: int,
                 in_channels: int,
                 num_filters: int = 2,
                 kernel_size: int = 3, 
                 stride: int = 1,
                 padding: int = 1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.conv = Conv2d(in_channels=in_channels, out_channels=num_filters, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = BatchNorm2d(num_features=num_filters)
        self.out = Linear(in_features=num_filters * input_size, out_features=output_size)
        

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = torch.nn.functional.relu(x)
        x = x.reshape(x.shape[0], -1)
        return self.out(x)