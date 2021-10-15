import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from CompositeLayer.nn import ConvPointConvolution
from CompositeLayer.nn.Conv.conv import CompositeConv
from CompositeLayer.nn.utils import apply_bn


class ADCompositeNet(nn.Module):

    def __init__(self, input_channels, output_channels, config, dimension=3):
        super(ADCompositeNet, self).__init__()

        self.neighbourhood = config["neighbours"]
        pl = config["pl"]  #32
        self.pl = pl
        self.rep_dim = output_channels
        # convolutions

        self.cv1 = CompositeConv(input_channels, pl, config, dimension, spatial_id=config["spatial"], semantic_id=config["semantic"])
        self.cv3 = CompositeConv(pl, 2 * pl,  config, dimension, spatial_id=config["spatial"], semantic_id=config["semantic"])
        self.cv4 = CompositeConv(2 * pl, 4 * pl, config, dimension, spatial_id=config["spatial"], semantic_id=config["semantic"])
        #self.cv5 = CompositeConv(4 * pl, 6 * pl, config,dimension, spatial_id=config["spatial"], semantic_id=config["semantic"])
        #self.cv6 = CompositeConv(6 * pl, 8 * pl, config, dimension, spatial_id=config["spatial"], semantic_id=config["semantic"])
        # last layer
        self.fcout = nn.Linear(4 * pl, output_channels, bias=False) # was 8*pl
        self.old_output_channels = output_channels
        self.fcout2 = nn.Linear(output_channels, 20, bias=False)
        # batchnorms
        self.bn1 = nn.BatchNorm1d(pl, eps=1e-4, affine=False, track_running_stats=False)
        self.bn3 = nn.BatchNorm1d(2 * pl, eps=1e-4, affine=False, track_running_stats=False)
        self.bn4 = nn.BatchNorm1d(4 * pl, eps=1e-4, affine=False, track_running_stats=False)
        #self.bn5 = nn.BatchNorm1d(6 * pl, eps=1e-4, affine=False, track_running_stats=False)
        #self.bn6 = nn.BatchNorm1d(8 * pl, eps=1e-4, affine=False, track_running_stats=False)

        self.dropout = nn.Dropout(config["dropout"])

        self.relu = nn.LeakyReLU(inplace=False, negative_slope=0.1)

    def forward(self, x, input_pts):

        x1, pts1 = self.cv1(x, input_pts, 32, 128)
        x1 = self.relu(x1)

        x3, pts3 = self.cv3(x1, pts1, 32, 32)
        x3 = self.relu(apply_bn(x3, self.bn3))

        x4, pts4 = self.cv4(x3, pts3, 32, 1)
        x4 = self.relu(x4)

        #x5, pts5 = self.cv5(x4, pts4, 16, 16)
        #x5 = self.relu(apply_bn(x5, self.bn5))

        #x6, _ = self.cv6(x5, pts5, 16, 1)
        #x6 = self.relu(x6)

        xout = x4.view(x4.size(0), -1)
        xout = self.dropout(xout)
        xout = self.fcout(xout)
        xreg = self.fcout2(self.relu(xout))


        return xout, xreg

    def getCenterParameters(self):
        layers = [
        self.cv1.getSpatialParams(),
        self.cv3.getSpatialParams(),
        self.cv4.getSpatialParams(),
        self.cv5.getSpatialParams(),
        self.cv6.getSpatialParams() ]

        params = [i for l in layers for i in l]
        return params

    def changeOutputLayer(self, output_channels):
        self.fcout2 = nn.Linear(self.old_output_channels, output_channels, bias=False)
        nn.init.kaiming_normal_(self.fcout.weight)
        self.rep_dim = output_channels


class ADConvPoint(nn.Module):

    def __init__(self, input_channels, output_channels, config, dimension=3):
        super(ADConvPoint, self).__init__()

        n_centers = config['num_weightpts']
        self.neighbourhood = config["neighbours"]
        pl = config["pl"]  #32
        self.pl = pl
        self.rep_dim = output_channels
        # convolutions



        self.cv1 = ConvPointConvolution(input_channels, pl, n_centers, dimension, use_bias=True)
        self.cv3 = ConvPointConvolution(pl, 2 * pl, n_centers, dimension, use_bias=True)
        self.cv4 = ConvPointConvolution(2 * pl, 4 * pl, n_centers, dimension, use_bias=True)
        self.cv5 = ConvPointConvolution(4 * pl, 4 * pl, n_centers, dimension, use_bias=True)
        self.cv6 = ConvPointConvolution(4 * pl, 8 * pl, n_centers, dimension, use_bias=True)

        # last layer
        self.fcout = nn.Linear(8 * pl, output_channels, bias=False) # was 8*pl
        self.old_output_channels = output_channels
        self.fcout2 = nn.Linear(output_channels, 20, bias=False) #20
        # batchnorms
        self.bn1 = nn.BatchNorm1d(pl, eps=1e-4, affine=False, track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(2 * pl, eps=1e-4, affine=False, track_running_stats=False)
        self.bn3 = nn.BatchNorm1d(2 * pl, eps=1e-4, affine=False, track_running_stats=False)
        self.bn4 = nn.BatchNorm1d(4 * pl, eps=1e-4, affine=False, track_running_stats=False)
        self.bn5 = nn.BatchNorm1d(6 * pl, eps=1e-4, affine=False, track_running_stats=False)
        self.bn6 = nn.BatchNorm1d(8 * pl, eps=1e-4, affine=False, track_running_stats=False)

        self.dropout = nn.Dropout(config["dropout"])

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, input_pts):

        x1, pts1 = self.cv1(x, input_pts, 32, 1024)
        x1 = self.relu(x1)

        x3, pts3 = self.cv3(x1, pts1, 32, 256)
        x3 = self.relu(apply_bn(x3, self.bn3))

        x4, pts4 = self.cv4(x3, pts3, 16, 64)
        x4 = self.relu(x4)

        x5, pts5 = self.cv5(x4, pts4, 16, 16)
        x5 = self.relu(apply_bn(x5, self.bn5))

        x6, _ = self.cv6(x5, pts5, 16, 1)
        x6 = self.relu(x6)

        xout = x6.view(x6.size(0), -1)
        xout = self.dropout(xout)
        xout = self.fcout(xout)
        xout_reg = self.fcout2(self.relu(xout))

        return xout, xout_reg

    def getCenterParameters(self):
        layers = [
        self.cv1.getSpatialParams(),
        self.cv3.getSpatialParams(),
        self.cv4.getSpatialParams(),
        self.cv5.getSpatialParams(),
        self.cv6.getSpatialParams() ]

        params = [i for l in layers for i in l]
        return params

    def changeOutputLayer(self, output_channels):
        self.fcout2 = nn.Linear(self.old_output_channels, output_channels, bias=False)
        nn.init.kaiming_normal_(self.fcout.weight)
        self.rep_dim = output_channels

