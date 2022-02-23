import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

#from global_tags import GlobalTags
#if GlobalTags.legacy_layer_base():
#    from ..legacy.layer_base import LayerBase
#else:
#    from ..layer_base import LayerBase
from CompositeLayer.nn.Conv import LayerBase

class RBFSpatialLayer(nn.Module):

    def __init__(self, config, dim, relu=False ):

        super(RBFSpatialLayer, self).__init__()

        self.dim = dim
        self.n_centers = config["n_centers"]
        self.spatial_function_dimension = config["spatial_function_dimension"]
        self.use_relu = relu
        # Weights
        self.spatial_weight = nn.Parameter(
                        torch.Tensor(self.n_centers, self.spatial_function_dimension).float(), requires_grad=True)
        nn.init.kaiming_normal_(self.spatial_weight.data, mode="fan_out")

        # centers
        center_data = np.zeros((dim, self.n_centers))
        for i in range(self.n_centers):
            coord = np.random.rand(dim)*2 - 1
            while (coord**2).sum() > 1:
                coord = np.random.rand(dim)*2 - 1
            center_data[:,i] = coord
        self.centers = nn.Parameter(torch.from_numpy(center_data).float(),
                                    requires_grad=True)

        self.relu = nn.LeakyReLU(inplace=True)

        self.bn = nn.BatchNorm1d(self.spatial_function_dimension)

    def rbf(self, dists_norm):
        #res = 1 - dists_norm.pow(2)
        res =torch.exp(-dists_norm.pow(2)/0.08) # was 0.09
        return res

    def forward(self, pts, K):


        # building the batched version of the summation index m
        dists = pts.unsqueeze(-1) - self.centers
        dists_norm = torch.norm(dists, dim=3)

        # first rbf layer
        spatial_layer = self.rbf(dists_norm)
        spatial_layer = spatial_layer @ self.spatial_weight

        if self.use_relu:
            spatial_layer = self.relu(spatial_layer)

        return spatial_layer

    def freeze(self):
        self.centers.requires_grad = False
        self.spatial_weight.requires_grad = False

    def unfreeze(self):
        self.centers.requires_grad = True
        self.spatial_weight.requires_grad = True

class TrainedRBFSpatialLayer(nn.Module):

    def __init__(self, config, dim, relu=False):

        super(TrainedRBFSpatialLayer, self).__init__()

        self.dim = dim
        self.n_centers = config["n_centers"]
        self.spatial_function_dimension = config["spatial_function_dimension"]
        self.use_relu = relu

        self.n_coefficients = 6
        self.pulse = 3.14

        self.weight_sin = nn.Parameter(torch.Tensor(
            self.n_centers,  # number of parameters used in the fourier expansion
            self.n_coefficients
        ).float(), requires_grad=True)  # .float().cuda()

        bound = math.sqrt(3.0) * math.sqrt(2.0 / (self.n_coefficients))
        self.weight_sin.data.uniform_(-bound, bound)

        self.weight_cos = nn.Parameter(torch.Tensor(
            self.n_centers,  # number of parameters used in the fourier expansion
            self.n_coefficients
        ).float(), requires_grad=True)  #
        self.weight_cos.data.uniform_(-bound, bound)

        self.expansion_index = (torch.arange(1, self.n_coefficients + 1).cuda() * self.pulse).float()
        self.expansion_index.requires_grad = False

        # Weights
        self.spatial_weight = nn.Parameter(
            torch.Tensor(self.n_centers, self.spatial_function_dimension).float(), requires_grad=True)
        self.spatial_weight.data.uniform_(-1, 1)
        self.expansion_index =  (torch.arange(1,self.n_coefficients + 1 ).cuda() * self.pulse).float()

        # centers
        center_data = np.zeros((dim, self.n_centers))
        for i in range(self.n_centers):
            coord = np.random.rand(dim) * 2 - 1
            while (coord ** 2).sum() > 1:
                coord = np.random.rand(dim) * 2 - 1
            center_data[:, i] = coord
        self.centers = nn.Parameter(torch.from_numpy(center_data).float(),
                                    requires_grad=True)

        self.relu = nn.ReLU(inplace=True)

        self.bn = nn.BatchNorm1d(self.spatial_function_dimension)

    def rbf(self, dists_norm):

        ds = dists_norm.shape
        dists_norm = dists_norm.view(-1,ds[3], 1 )
        indexes = self.expansion_index.view(1, self.n_coefficients)

        args = (dists_norm @ indexes)

        cos_terms = torch.cos(args)*self.weight_cos
        sin_terms = torch.sin(args)*self.weight_sin

        expansion = (cos_terms + sin_terms).sum(2)

        return expansion.view(ds)

    def forward(self, pts, K):

        # building the batched version of the summation index m
        dists = pts.unsqueeze(-1) - self.centers
        dists_norm = torch.norm(dists, dim=3)

        # first rbf layer
        spatial_layer = self.rbf(dists_norm)
        spatial_layer = spatial_layer @ self.spatial_weight

        if self.use_relu:
            spatial_layer = self.relu(spatial_layer)

        return spatial_layer
    def freeze(self):
        self.weight_sin.requires_grad = False
        self.weight_cos.requires_grad = False
        self.centers.requires_grad = False
        self.spatial_weight.requires_grad = False

    def unfreeze(self):
        self.weight_sin.requires_grad = True
        self.weight_cos.requires_grad = True
        self.centers.requires_grad = True
        self.spatial_weight.requires_grad = True


class ConvpointSpatialLayer(nn.Module):

    def __init__(self, config, dim, relu=False):

        super(ConvpointSpatialLayer, self).__init__()

        self.dim = dim
        self.n_centers = config["n_centers"]
        self.spatial_function_dimension = config["spatial_function_dimension"]
        self.use_relu = relu

        # Weights
        self.l1 = nn.Linear(dim*self.n_centers, 2*self.n_centers, bias=config["biases"])
        self.l2 = nn.Linear(2*self.n_centers, self.n_centers, bias=config["biases"])
        self.l3 = nn.Linear(self.n_centers, self.spatial_function_dimension, bias=config["biases"])

        # centers
        center_data = np.zeros((dim, self.n_centers))
        for i in range(self.n_centers):
            coord = np.random.rand(dim) * 2 - 1
            while (coord ** 2).sum() > 1:
                coord = np.random.rand(dim) * 2 - 1
            center_data[:, i] = coord
        self.centers = nn.Parameter(torch.from_numpy(center_data).float(),
                                    requires_grad=True)

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, pts, K):

        BATCH_SIZE = pts.size(0)
        PTS_PER_POINT_CLOUD = pts.size(1)
        NEIGHBOURS = K

        # building the batched version of the summation index m
        dists = (pts.unsqueeze(-1) - self.centers)
        dists = dists.view(BATCH_SIZE, PTS_PER_POINT_CLOUD, NEIGHBOURS, -1)

        dists = self.relu(self.l1(dists))
        dists = self.relu(self.l2(dists))
        dists = self.relu(self.l3(dists))

        spatial_layer = dists

        return spatial_layer

