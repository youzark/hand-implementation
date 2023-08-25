import torch
from torch import nn, std

class BasicConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, activation = nn.ReLU()):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = kernel_size // 2
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.convolution(x)
        x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x
    
class ResNetEmbedding(nn.Module):
    """
    batch data : (Batch, Weight, Height, Channel) ->
    Embedding  : (Batch, Weight, Height, Feature Dimension)
    with a convolutional layer and a max pooling
    """
    def __init__(self, image_channels, hidden_channels):
        super().__init__()
        self.convolution = BasicConvLayer(
            in_channels=image_channels,
            out_channels= hidden_channels,
            kernel_size= 7,
            stride= 2
        )
        self.maxpool = nn.MaxPool2d(
            kernel_size= 3,
            stride= 2,
            padding= 1
        )

    def forward(self, x):
        x = self.convolution(x)
        x = self.maxpool(x)
        return x

class ResNetShortCuts(nn.Module):
    """
    reshape Layer input data to match the output data
    so that we can perform residual connection in that layer
    """
    def __init__(self, in_channels, out_channels, stride = 1):
        self.need_projection = (in_channels != out_channels) or (stride != 1)
        self.conv = BasicConvLayer(
            in_channels= in_channels,
            out_channels= out_channels,
            kernel_size= 1,
            stride= stride
        )

    def forward(self, x):
        if self.need_projection:
            x = self.conv(x)
        return x

class ResNetBottleNeckLayer(nn.Module):
    """
    Building block of ResNet Model:
    Downscale feature dimension before doing 3*3 convlution to save computation.
    Implement residual connection with feature dimension and feature map dimension taking into account.
    """
    def __init__(self, in_channels, out_channels, stride = 1, downscale_factor = 4, activation = nn.ReLU()):
        super().__init__()
        self.shortcut = ResNetShortCuts(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride
        )
        self.reduced_channels = out_channels // downscale_factor
        self.conv = nn.Sequential(
            BasicConvLayer(
                in_channels=in_channels,
                out_channels=self.reduced_channels,
                kernel_size=1
            ),
            BasicConvLayer(
                in_channels=self.reduced_channels,
                out_channels=self.reduced_channels,
                kernel_size=3,
                stride = stride
            ),
            BasicConvLayer(
                in_channels=self.reduced_channels,
                out_channels=out_channels,
                kernel_size=1
            ),
        )
        self.activation = activation

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv(x)
        x += identity
        x = self.activation(x)
        return x

class ResNetBottleNeckStack(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            stride = 2,
            depth = 2,
        ):
        self.layers = nn.Sequential(
            ResNetBottleNeckLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
            ),
            *[ ResNetBottleNeckLayer(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
            ) for _ in range(depth - 1)
            ],
        )




