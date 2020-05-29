import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, fixed_dims, num_dilation):
        super(ResBlock, self).__init__()

        self.conv_1 = nn.Sequential(nn.Conv2d(fixed_dims, fixed_dims, kernel_size=3, stride=1, padding=num_dilation, dilation=num_dilation, bias=False),
                                    nn.BatchNorm2d(fixed_dims),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.conv_2 = nn.Sequential(nn.Conv2d(fixed_dims, fixed_dims, kernel_size=3, stride=1, padding=num_dilation, dilation=num_dilation, bias=False),
                                    nn.BatchNorm2d(fixed_dims))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        residual_part = self.conv_1(x)
        residual_part = self.conv_2(residual_part)
        output = self.leaky_relu(x + residual_part)
        return output

class SmoothNet(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(SmoothNet, self).__init__()

        self.conv_1 = nn.Sequential(nn.Conv2d(input_dims, 16, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(16),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv_2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(32),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv_3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.res_block_1 = ResBlock(64, 2)
        self.res_block_2 = ResBlock(64, 2)
        self.res_block_3 = ResBlock(64, 4)
        self.res_block_4 = ResBlock(64, 4)
        self.res_block_5 = ResBlock(64, 8)
        self.res_block_6 = ResBlock(64, 8)
        self.res_block_7 = ResBlock(64, 16)
        self.res_block_8 = ResBlock(64, 16)
        self.res_block_9 = ResBlock(64, 1)
        self.res_block_10 = ResBlock(64, 1)

        self.de_conv_4 = nn.Sequential(nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.LeakyReLU(negative_slope=0.2))
        self.conv_5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv_6 = nn.Sequential(nn.Conv2d(64, 3, kernel_size=1, stride=1, bias=False))

    def forward(self, x):
        original = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = self.res_block_3(x)
        x = self.res_block_4(x)
        x = self.res_block_5(x)
        x = self.res_block_6(x)
        x = self.res_block_7(x)
        x = self.res_block_8(x)
        x = self.res_block_9(x)
        x = self.res_block_10(x)
        x = self.de_conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        output = original + x
        return output