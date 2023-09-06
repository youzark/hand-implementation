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
                 stride : int = 1, # stride here basically mean upsampling factor in deconvolution
                 output_padding: int = 0,
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
            output_padding = output_padding,
            bias= True
        )

    def forward(self, x):
        if self.need_short_cut:
            x = self.deconv(x)
        return x

class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        output_padding: int = 0,
        ):
        super().__init__()
        need_upsampling = (stride > 1)
        self.layers = nn.Sequential(
            BasicConvBlock(
                in_channels = in_channels,
                out_channels = out_channels,
                stride = stride,
            ) if not need_upsampling 
            else nn.ConvTranspose2d(
                in_channels= in_channels,
                out_channels= out_channels,
                kernel_size= 3,
                padding= 1,
                stride = stride,
                output_padding = output_padding,
            ),
            BasicConvBlock(
                in_channels = out_channels,
                out_channels = out_channels,
                activation= None
            ),
        )
        self.shortcut = ShortCut(
            in_channels= in_channels,
            out_channels= out_channels,
            stride= stride,
            output_padding= output_padding
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.layers(x)
        x += self.shortcut(identity)
        x = self.activation(x)
        return x

class BottleNeckStack(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        depth:int,
        output_padding: int = 0,
        ):
        super().__init__()
        self.layers = nn.Sequential(
            ResBlock(
                in_channels = in_channels,
                out_channels= out_channels,
                stride= stride,
                output_padding= output_padding
            ),
            *[
                ResBlock(
                    in_channels = out_channels,
                    out_channels= out_channels,
                    stride = 1
                ) 
                for _ in range(depth-1)
            ],
        )

    def forward(self,x):
        x = self.layers(x)
        return x


class Decoder(nn.Module):
    """
    ResNet Decoder that recover 2D image from feature map:
    (Batch_size, Channel_feature_map, Height_feature_map, Width_feature_map) =>
    (Batch_size, Channel_image, Height_image, Width_image)

    config defines feature dimension and (width,height) scaling per layer and repetition per layer:
    {
        "stack_configs": [
            {
                "in_channels":int,
                "out_channels":int,
                "stride":int, # control the scaling of feature map,
                "depth":int, # control repetition of basic BottleNeckLayer in the stack
                "output_padding":int, # for rounding might happen in Encoder phase(when the input dim is even), the size of feature map before down-sample and after up-sample might mismatch, a corresponding output_padding must be introduced to counter effect rounding in Encoder Phase.
            },
            {
                "in_channels":int,
                "out_channels":int,
                "stride":int, # control the scaling of feature map,
                "depth":int, # control repetition of basic BottleNeckLayer in the stack
                "output_padding":int, # for rounding might happen in Encoder phase(when the input dim is even), the size of feature map before down-sample and after up-sample might mismatch, a corresponding output_padding must be introduced to counter effect rounding in Encoder Phase.
            },
        ]
    }


    !!! further explaination on encoder decoder image size alignment:
    In this auto-encoder architecture, we must have same input output image size to properly calculate reconstruction error.
    Ignoring dilation, in encoding phase, the up-sampling formulation is:
    H_down_out = floor((H_down_in + 2*padding_down - 1)/stride_down + 1)
    in Decoding phase:
    H_up_out = (H_up_in - 1)*stride_up - 2*padding_up + output_padding + 1

    Assume:
    stride_up = stride_down = 2

    fed the output of encoder to corresponding decoder layer:
    H_up_in = H_down_out
    We get:
    H_up_out = 2*floor((H_down_in + 1) / 2) + output_padding - 1
    
    So to align input to encoder and output of decoder:( <=> H_up_out == H_down_in)
    output_padding = 0 if H_down_in%2 == 1 else 1
    """
    def __init__(self, model_config,):
        super().__init__()
        self.layers = nn.Sequential(*[
            BottleNeckStack(
                in_channels= layer["in_channels"],
                out_channels= layer["out_channels"],
                stride= layer["stride"],
                output_padding = layer["output_padding"],
                depth = layer["depth"],
            )
            for layer in model_config["stack_configs"]
        ])


    def forward(self, feature_map):
        image = self.layers(feature_map)
        return image
