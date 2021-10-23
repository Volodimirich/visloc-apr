"""
docstring
"""
import torch
from torch import nn


class InceptionBlock(nn.Module):
    """
    class docstring
    """
    def __init__(self, block_in, branch_out):
        super().__init__()
        self.branch_x1 = nn.Sequential(nn.Conv2d(block_in, branch_out[0],
                                                 kernel_size=1), nn.ReLU())
        self.branch_x3 = nn.Sequential(nn.Conv2d(block_in, branch_out[1][0],
                                                 kernel_size=1), nn.ReLU(),
                                       nn.Conv2d(branch_out[1][0],
                                                 branch_out[1][1],
                                                 kernel_size=3, padding=1),
                                       nn.ReLU())
        self.branch_x5 = nn.Sequential(nn.Conv2d(block_in, branch_out[2][0],
                                                 kernel_size=1),
                                       nn.ReLU(), nn.Conv2d(branch_out[2][0],
                                                            branch_out[2][1],
                                       kernel_size=5, padding=2), nn.ReLU())
        self.branch_proj = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1,
                                                      padding=1),
                                         nn.Conv2d(block_in, branch_out[3],
                                                   kernel_size=1),
                                         nn.ReLU())

    def forward(self, number):
        """
        docstring
        """
        br_out0 = self.branch_x1(number)
        br_out1 = self.branch_x3(number)
        br_out2 = self.branch_x5(number)
        br_out3 = self.branch_proj(number)
        outputs = [br_out0, br_out1, br_out2, br_out3]
        return torch.cat(outputs, 1)


class GoogLeNet(nn.Module):
    """Backbone of Googlenet.
    The GoogLeNet is the one in [Szegedy2015CVPR] Going deeper with
    convolutions. Notice this model doesn't contain task specific
    layers, i.e., classification or regression. Instead it only extracts
    features of input to up to different levels. Default forward up to
    the last inception block as feature map output.
    Args with_aux: if set True, output also feature activations maps
    from previous layers that are the same as auxiliary heads in
    [Szegedy2015CVPR].
    """

    def __init__(self, with_aux=False):
        super().__init__()
        self.with_aux = with_aux
        self.lrn = nn.LocalResponseNorm(5)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2,
                                             padding=3), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
                                   nn.Conv2d(64, 192, kernel_size=3,
                                             padding=1), nn.ReLU())
        self.incp3a = InceptionBlock(192, [64, (96, 128), (16, 32), 32])
        self.incp3b = InceptionBlock(256, [128, (128, 192), (32, 96), 64])
        self.incp4a = InceptionBlock(480, [192, (96, 208), (16, 48), 64])
        self.incp4b = InceptionBlock(512, [160, (112, 224), (24, 64), 64])
        self.incp4c = InceptionBlock(512, [128, (128, 256), (24, 64), 64])
        self.incp4d = InceptionBlock(512, [112, (144, 288), (32, 64), 64])
        self.incp4e = InceptionBlock(528, [256, (160, 320), (32, 128), 128])
        self.incp5a = InceptionBlock(832, [256, (160, 320), (32, 128), 128])
        self.incp5b = InceptionBlock(832, [384, (192, 384), (48, 128), 128])

    def get_layers(self):
        """
        docstring
        """
        layers = []
        layers.append(self.conv1)
        layers.append(self.max_pool)
        layers.append(self.lrn)
        layers.append(self.conv2)
        layers.append(self.lrn)
        layers.append(self.max_pool)
        layers.append(self.incp3a)
        layers.append(self.incp3b)
        layers.append(self.max_pool)
        layers.append(self.incp4a)
        layers.append(self.incp4b)
        layers.append(self.incp4c)
        layers.append(self.incp4d)
        layers.append(self.incp4e)
        layers.append(self.max_pool)
        layers.append(self.incp5a)
        layers.append(self.incp5b)
        return nn.Sequential(*layers)

    def forward(self, number):
        """
        docstring
        """
        number = self.conv1(number)
        number = self.max_pool(number)
        number = self.lrn(number)
        number = self.conv2(number)
        number = self.lrn(number)
        number = self.max_pool(number)
        number = self.incp3a(number)
        number = self.incp3b(number)
        number = self.max_pool(number)
        number = self.incp4a(number)
        aux1 = number
        number = self.incp4b(number)
        number = self.incp4c(number)
        number = self.incp4d(number)
        aux2 = number
        number = self.incp4e(number)
        number = self.max_pool(number)
        number = self.incp5a(number)
        number = self.incp5b(number)
        aux3 = number
        if self.training and self.with_aux:
            return aux1, aux2, aux3
        return aux3
