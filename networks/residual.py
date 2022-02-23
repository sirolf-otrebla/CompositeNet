import torch
import torch.nn as nn
import torch.nn.functional as F
from CompositeLayer.nn import ConvPointConvolution
from CompositeLayer.nn.Conv.residual import ResnetBlock
from CompositeLayer.nn.utils import apply_bn
from CompositeLayer.nn.Conv.conv import CompositeConv



class ResidualCompositeNet(nn.Module):

    def __init__(self, input_channels, output_channels, config, dimension=3):
        super(ResidualCompositeNet, self).__init__()

        n_centers = 16
        expansion_length = 8
        pl = config["pl"]

        # convolutions




        self.cv1 = ResnetBlock(input_channels, pl, config, dimension, config["spatial"], config["semantic"])
        self.cv3 = ResnetBlock(pl, 2 * pl,  config,dimension, config["spatial"], config["semantic"])
        self.cv4 = ResnetBlock(2 * pl, 4 * pl, config, dimension, config["spatial"], config["semantic"])
        self.cv5 = ResnetBlock(4 * pl, 8 * pl, config,dimension, config["spatial"], config["semantic"])
        self.cv6 = ResnetBlock(8 * pl, 16 * pl, config, dimension, config["spatial"], config["semantic"])
        # last layer
        self.fcout = nn.Linear(16* pl, output_channels)

        # batchnorms
        self.bn1 = nn.BatchNorm1d(pl)
        #self.bn2 = nn.BatchNorm1d(2 * pl)
        self.bn3 = nn.BatchNorm1d(2 * pl)
        self.bn4 = nn.BatchNorm1d(4 * pl)
        self.bn5 = nn.BatchNorm1d(8* pl)
        self.bn6 = nn.BatchNorm1d(16 * pl)

        self.dropout = nn.Dropout(config["dropout"])

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, input_pts):

        x1, pts1 = self.cv1(x, input_pts, 32, 1024)
        x1 = self.relu(apply_bn(x1, self.bn1))

        x3, pts3 = self.cv3(x1, pts1, 32, 256)
        x3 = self.relu(apply_bn(x3, self.bn3))

        x4, pts4 = self.cv4(x3, pts3, 16, 64)
        x4 = self.relu(apply_bn(x4, self.bn4))

        x5, pts5 = self.cv5(x4, pts4, 16, 16)
        x5 = self.relu(apply_bn(x5, self.bn5))

        x6, _ = self.cv6(x5, pts5, 16, 1)
        x6 = self.relu(apply_bn(x6, self.bn6))

        xout = x6.view(x6.size(0), -1)
        xout = self.dropout(xout)
        xout = self.fcout(xout)

        return xout

    def getCenterParameters(self):
        return 1


