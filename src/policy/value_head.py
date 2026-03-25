import torch
from torch.nn import Sequential, Linear, Conv2d, BatchNorm2d, ReLU, Tanh

class ValueHead(torch.nn.Module):
    def __init__(self, 
                 input_size: int,
                 in_channels: int,
                 hidden_size: int = 256,
                 num_filters: int = 1,
                 kernel_size: int = 3, 
                 stride: int = 1,
                 padding: int = 1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.conv = Conv2d(in_channels=in_channels, out_channels=num_filters, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = BatchNorm2d(num_features=num_filters)
        self.hidden = Linear(in_features=num_filters * input_size, out_features=hidden_size)
        self.out = Linear(in_features=hidden_size, out_features=1)
        

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = torch.nn.functional.relu(x)
        x = x.reshape(x.shape[0], -1)
        x = self.hidden(x)
        x = torch.nn.functional.relu(x)
        return self.out(x)