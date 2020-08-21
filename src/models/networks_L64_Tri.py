import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

class ResnetConditionTriGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        self.n_blocks = n_blocks
        self.div = 3
        super(ResnetConditionTriGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model1 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model1 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        con_dim = 16
        model2 = []

        for i in range(n_blocks):       # add ResNet blocks
            if (i + 1) % self.div == 0:
                model2 += [
                    ResnetBlock(ngf * mult + con_dim, ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                use_dropout=use_dropout, use_bias=use_bias)]
            else:
                model2 += [ResnetBlock(ngf * mult, ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                       use_dropout=use_dropout, use_bias=use_bias)]

        model3 = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model3 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model3 += [nn.ReflectionPad2d(3)]
        model3 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model3 += [nn.Tanh()]
        model_landmark_trans = [nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=use_bias), norm_layer(8), nn.ReLU(True),
                                nn.Conv2d(8, con_dim, kernel_size=3, padding=1, bias=use_bias), norm_layer(con_dim), nn.ReLU(True),
                                nn.Conv2d(con_dim, con_dim, kernel_size=3, padding=1, bias=use_bias), norm_layer(con_dim)]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model_landmark_trans = nn.Sequential(*model_landmark_trans)

    def forward(self, input, land2):
        """Standard forward"""
        x = self.model1(input)
        l2 = self.model_landmark_trans(land2)
        for i in range(self.n_blocks):
            if (i + 1) % self.div == 0:
                x = self.model2[i](torch.cat([x, l2], dim=1))
            else:
                x = self.model2[i](x)
        out = self.model3(x)

        return out


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim_in, dim_out, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim_in, dim_out, padding_type, norm_layer, use_dropout, use_bias)
        self.shortcut = nn.Sequential(*[nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1, bias=use_bias), norm_layer(dim_out)])

    def build_conv_block(self, dim_in, dim_out, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim_out), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim_out)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = self.shortcut(x) + self.conv_block(x)  # add skip connections
        return out
