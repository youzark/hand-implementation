from torch import nn
import torch
import torch.functional as F
import torchvision
from torchvision.utils import save_image

class Bottleneck(nn.Module):
    """
    Bottleneck design to train deeper network:
    reduce parameter with kernel size (1,1) for downscaling and upscaling features at the begin and end of the the block
    input_channels => input_channels // 2 => input_channels * 2
    (but the first layer doesn't do this dowscaling)

    we define: 
    working_channels = out_channels // 4
    """
    def __init__(self, in_channels, out_channels, identity_downsample=None,
                 stride=1):
        super().__init__()
        working_channels = out_channels // 4
        self.conv1 = nn.Conv2d(in_channels, working_channels, 
                               kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(working_channels)

        self.conv2 = nn.Conv2d(working_channels, working_channels, kernel_size=3,
                               stride=stride, padding= 1)
        self.bn2 = nn.BatchNorm2d(working_channels)

        self.conv3 = nn.Conv2d(working_channels, out_channels,
                               kernel_size= 1, stride= 1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.identity_downsample = identity_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.identity_downsample:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)

        return x

class Block(nn.Module):
    """
    for smaller network, there is no need to downsample and upsample channels before and after doing 3,3 convolution calculation.
    """
    def __init__(self, in_channels, out_channels, identity_downsample = None, 
                 stride = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               padding= 1, stride= stride, bias= False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding= 1, stride= stride, bias= False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.identity_downsample = identity_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.identity_downsample:
            identity = self.identity_downsample(identity)

        x = x + identity
        x = self.relu()

        return x
    

class ResNet(nn.Module):
    def __init__(self, layer_list, classes):
        """
        layer_list = [
            {
                "in_channels": int,
                "out_channels": int,
                "repetition": int,
                "stride": int
            }
        ]
        specify in_channels, out_channels of the whole layer and how many Bottlenecks in the layer and the stride for the first Bottleneck in the layer
        """
        super().__init__()

        image_channels = 3 # R,G,B
        init_channels = layer_list[0]["in_channels"]
        self.conv1 = nn.Conv2d(image_channels, init_channels, kernel_size= 7,
                               stride= 2, padding= 3, bias= False)
        self.bn1 = nn.BatchNorm2d(init_channels)
        self.relu = nn.ReLU()

        self.max_pool = nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1)

        self.conv_layers = nn.Sequential(*[
            self._make_layer(layer["in_channels"],layer["out_channels"],
                             layer["repetition"],stride= layer["stride"])
            for layer in layer_list
        ])

        self.avgpool = nn.AdaptiveMaxPool2d((1,1))

        self.fc = nn.Linear(layer_list[-1]["out_channels"], classes)


    def forward(self, x):
        x = self.max_pool(self.relu(self.bn1(self.conv1(x))))

        x = self.conv_layers(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0],-1)
        logits = self.fc(x)

        return logits


    def _make_layer(self, in_channels, out_channels, repetition, stride = 1):
        """
        stack ${repetition} Bottleneck(or Block) to form a layer.
        the first layer is responsible for downsample the image dimensions(not the channels) by taking stride not equal to 2
        """
        identity_downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels)
        )
        layers = [
            Bottleneck(in_channels, out_channels, identity_downsample=identity_downsample, 
                       stride=stride)
        ]
        for _ in range(repetition - 1):
            layers.append(Bottleneck(out_channels,out_channels))

        return nn.Sequential(*layers)
