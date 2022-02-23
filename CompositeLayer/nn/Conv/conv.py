import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from CompositeLayer.nn.Conv.feature_sublayers import *
from CompositeLayer.nn.Conv.spatial_sublayers import *

from global_tags import GlobalTags
if GlobalTags.legacy_layer_base():
    from ..legacy.layer_base import LayerBase
else:
    from ..layer_base import LayerBase


class ConvPointConvolution(LayerBase):
    def __init__(self, input_features, output_features, n_centers, dim, use_bias=True):
        super(ConvPointConvolution, self).__init__()

        # Weight
        self.weight = nn.Parameter(
                        torch.Tensor(input_features, n_centers, output_features), requires_grad=True)
        bound = math.sqrt(3.0) * math.sqrt(2.0 / (input_features + output_features))
        self.weight.data.uniform_(-bound, bound)

        # bias
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_features), requires_grad=True)
            self.bias.data.uniform_(-bound,bound)

        # centers
        center_data = np.zeros((dim, n_centers))
        for i in range(n_centers):
            coord = np.random.rand(dim)*2 - 1
            while (coord**2).sum() > 1:
                coord = np.random.rand(dim)*2 - 1
            center_data[:,i] = coord
        self.centers = nn.Parameter(torch.from_numpy(center_data).double(),
                                    requires_grad=True)

        # MLP 2034 params
        self.l1 = nn.Linear(dim*n_centers, 2*n_centers)
        self.l2 = nn.Linear(2*n_centers, n_centers)
        self.l3 = nn.Linear(n_centers, n_centers)



    # input contains qualitative features ( BATCH_SIZE x   NUM_POINTS x FEATURES )
    # points contains spatial features (  BATCH_SIZE x  NUM_POINTS x SPATIAL_DIM )
    def forward(self, input, points, K, next_pts=None, normalize=True, indices_=None, return_indices=False, dilation=1):
        # debug
        input_shape = input.shape
        points_shape = points.shape

        if indices_ is None:
            if isinstance(next_pts, int) and points.size(1) != next_pts:
                # convolution with reduction
                indices, next_pts_ = self.indices_conv_reduction(points, K * dilation, next_pts)
            elif (next_pts is None) or (isinstance(next_pts, int) and points.size(1) == next_pts):
                # convolution without reduction
                # we have a matrix indices where indices[i] is the neighborhood of pt i
                # the second return value is simply the point list as previously passed as argument,
                #  so that the operation can be repeated. if stride != 0, the second argument would be smaller in size
                indices, next_pts_ = self.indices_conv(points, K * dilation)
            else:
                # convolution with up sampling or projection on given points
                indices, next_pts_ = self.indices_deconv(points, next_pts, K * dilation)

            if next_pts is None or isinstance(next_pts, int):
                next_pts = next_pts_

            if return_indices:
                indices_ = indices
        else:
            indices = indices_
        # here indices should have this form
        # ( NUMBER OF POINT CLOUDS  x NUMBER OF pts x NUMBER OF NEIGHBOURS K  )

        batch_size = input.size(0)
        n_pts = input.size(1)

        if dilation > 1:
            indices = indices[:,:, torch.randperm(indices.size(2))]
            indices = indices[:,:,:K]

        # compute indices for indexing points
        # if clouds are 16 with 1024 pts each , add_indices would be 1*1024, 2*1024, 3*1024, ... 16*1024
        # indices now will have all indices from 1 to 16*1024
        # the batch will run "in parallel" during the same forward() execution
        add_indices = torch.arange(batch_size).type(indices.type()) * n_pts
        # it is necessary to arrange add_indices to be compatible with indices dimension
        # note that size -1 is inferred from other dimensions
        # indices becomes a vector of ( NUMBER OF POINT CLOUDS  x NUMBER OF pts x NUMBER OF NEIGHBOURS K  )
        indices = indices + add_indices.view(-1,1,1)

        # get the features and point cooridnates associated with the indices
        # we ptoduce two vectors with the size ( BATCH_SIZE x N_POINTS x K neighbours x FEATURE NUMBER )
        features_view = input.view(-1, input.size(2))
        features = features_view[indices]
        pts_view = points.view(-1, points.size(2))
        pts = pts_view[indices]
        pts_old = pts

        # center the neighborhoods
        # points is now a long vector of points, while next_pts was a subset of the original points
        # next_pts.unsqueeze(2) adds a dimension to (  BATCH_SIZE x  NUM_POINTS x SPATIAL_DIM )
        # nets_pts now becomes (  BATCH_SIZE x  NUM_POINTS x 1 x SPATIAL_DIM )
        # obtain neighbour pts shifted so that the corresponding output is the Origin
        # pts being (BATCH_SIZE x POINT_PER_CLOUD x NEIGHBOUR x SPATIAL_DIM
        next_pts_unsqueezed = next_pts.unsqueeze(2)
        pts = pts - next_pts_unsqueezed

        # normalize to unit ball, or not
        if normalize:
            maxi = torch.sqrt((pts.detach()**2).sum(3).max(2)[0]) # detach is a modificaiton
            maxi[maxi==0] = 1
            pts = pts / maxi.view(maxi.size()+(1,1,))

        # compute the distances
        dists = pts.view(pts.size()+(1,)) - self.centers
        dists = dists.view(dists.size(0), dists.size(1), dists.size(2), -1)
        dists = F.relu(self.l1(dists))
        dists = F.relu(self.l2(dists))
        dists = F.relu(self.l3(dists))

         # compute features
        fs = features.size()
        features = features.transpose(2,3)
        features = features.view(-1, features.size(2), features.size(3))
        dists = dists.view(-1, dists.size(2), dists.size(3))

        features = torch.bmm(features, dists)

        features = features.view(fs[0], fs[1], -1)

        features = torch.matmul(features, self.weight.view(-1, self.weight.size(2)))
        features = features/ fs[2]

        # add a bias
        if self.use_bias:
            features = features + self.bias

        if return_indices:
            return features, next_pts, indices_
        else:
            return features, next_pts


class CompositeConv(LayerBase):
    # 75 aa 81 OA
    def __init__(self, input_features, output_features, config, dim, spatial_id, semantic_id ):

        super(CompositeConv, self).__init__()

        use_bias = config["biases"]
        self.dim = dim
        self.input_features = input_features
        self.output_features = output_features

        self.spatial_sublayer = self.build_spatial(config, dim, spatial_id)
        self.feature_sublayer = self.build_semantic(input_features, output_features, config, semantic_id, use_bias=use_bias)


    def build_spatial(self, config, dim, id):
        if id == "RBFN-relu":
            return  RBFSpatialLayer(config, dim, relu=True)
        elif id == "RBFN-norelu":
            return RBFSpatialLayer(config, dim, relu=False)
        elif id == "TRBFN-relu":
            return TrainedRBFSpatialLayer(config, dim, relu=True)
        elif id == "TRBFN-norelu":
            return TrainedRBFSpatialLayer(config, dim, relu=False)
        elif id == "Convpoint":
            return ConvpointSpatialLayer(config, dim)
        return

    def build_semantic(self, input_features, output_features, config, id,  use_bias=False):
        if id == "linear":
            return  LinearSemanticLayer(input_features, output_features, config, relu=False, use_bias=use_bias)
        elif id == "aggregate":
            return AggregateSemanticLayer(input_features, output_features, config, relu=False, use_bias=use_bias)
        elif id == "MLP":
            return MLPSemanticLayer(input_features, output_features, config, relu=False, use_bias=use_bias)
        return

    def forward(self, input, points, K, next_pts=None, normalize=True, indices_=None, return_indices=False, dilation=1):

        pts, features, next_pts, indices_ = self.points_processing(input, points, K, next_pts, normalize, indices_, return_indices, dilation)

        out = self.spatial_sublayer(pts, K)
        out = self.feature_sublayer(features, out, K)

        if return_indices:
            return out, next_pts, indices_
        else:
            return out, next_pts
