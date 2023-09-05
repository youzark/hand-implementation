from torch import nn
from torch.nn import ReLU
from typing import Tuple

class BasicConvBlock(nn.Module):
    """
    Wrap up for {Convolution, Normalization and activation(optional)}
    """
    def __init__(self,
                 in_channels : int, 
                 out_channels : int, 
                 stride : int = 1,
                 kernel_size : int = 3,
                 activation : ReLU | None = nn.ReLU()
                 ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = kernel_size // 2
        )
        self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.normalization(x)
        if self.activation:
            x = self.activation(x)
        return x

class ShortCut(nn.Module):
    """
    Block used to Align input shape and output shape in Residual Connection
    """
    def __init__(self,
                 in_channels : int,
                 out_channels : int,
                 stride : int = 1 # stride here basically mean upsampling factor in deconvolution
                 ):
        """
        if in_channels and out_channels mismatch
        or stride not equal to 1  
        we need some transformation to the input to match the shape of the output in residual connection
        """
        super().__init__()
        self.need_short_cut = (stride > 1) or (in_channels != out_channels)
        self.deconv = nn.ConvTranspose2d(
            in_channels= in_channels,
            out_channels = out_channels,
            kernel_size = 1,
            stride= stride,
            bias= True
        )

    def forward(self, x):
        if self.need_short_cut:
            x = self.deconv(x)
        return x

class BottleNeckBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int,int],
        stride: int = 1,
        activation: nn.ReLU | None = None,
        ):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels = in_channels,
            out_channels= out_channels,
            kernel_size= kernel_size,
            stride = stride,
            padding= kernel_size // 2 if isinstance(kernel_size,int) else (kernel_size[0]//2,kernel_size[1]//2),   # to align with encoder parameter
        )
        self.activation = activation
        self.normalization = nn.ReLU()

    def forward(self, x):
        x = self.deconv(x)
        x= self.normalization(x)
        if self.activation:
            x = self.activation(x)
        return x

class BottleNeckStack(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        ):
        self.layers = nn.Sequential(
            BottleNeckBlock(
                in_channels = in_channels,
                out_channels= out_channels,
                kernel_size= 1,
                activation = nn.ReLU(),
            ),
            BottleNeckBlock(
                in_channels = out_channels,
                out_channels= out_channels,
                kernel_size= 3,
                stride= stride,
                activation = nn.ReLU(),
            ),
            BottleNeckBlock(
                in_channels = out_channels,
                out_channels= out_channels,
                kernel_size= 1,
                activation = None,
            ),
        )
        self.activation = nn.ReLU()
        self.shortcut = ShortCut(
            in_channels = in_channels,
            out_channels= out_channels,
            stride= stride
        )

        def forward(self,x):
            x = self.layers(x)
            x += self.shortcut(x.clone())
            return self.activation(x)


class Decoder(nn.Module):
    """
    ResNet Decoder that recover 2D image from feature map:
    (Batch_size, Channel_feature_map, Height_feature_map, Width_feature_map) =>
    (Batch_size, Channel_image, Height_image, Width_image)

    config defines feature dimension and (width,height) scaling per layer and repetition per layer:
    {
        "image_channels":int,
        "image_width":int,
        "image_height":int,
        "stack_configs": [
            {
                "in_channels":int,
                "out_channels":int,
                "stride":int, # control the scaling of feature map,
                "depth":int, # control repetition of basic BottleNeckLayer in the stack
            },
            {
                "in_channels":int,
                "out_channels":int,
                "stride":int, # control the scaling of feature map,
                "depth":int, # control repetition of basic BottleNeckLayer in the stack
            },
        ]
    }
    """
    def __init__(self, model_config,):
        super().__init__()
        self.layers = nn.Sequential(*[
            BottleNeckStack(
                in_channels= layer["in_channels"],
                out_channels= layer["out_channels"],
                stride= layer["stride"],
            )
            for layer in model_config["stack_configs"]
        ])


    def forward(self, feature_map):
        image = self.layers(feature_map)
        return image
