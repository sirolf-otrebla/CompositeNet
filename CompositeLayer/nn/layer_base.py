import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import CompositeLayer.knn.lib.python.nearest_neighbors as nearest_neighbors


class LayerBase(nn.Module):

    def __init__(self):
        super(LayerBase, self).__init__()

    def indices_conv_reduction(self, input_pts, K, npts):

        # I modified this code so that we do not consider the pt being in the center of our neighbourhood
        # this in order to avoid a "bias-like" effect in AD with SVDD
        indices, queries = nearest_neighbors.knn_batch_distance_pick(input_pts.cpu().detach().numpy(), npts, K, omp=True)
        indices = torch.from_numpy(indices).long()
        queries = torch.from_numpy(queries).float()

        # TODO need to put some flag here
        # if True:
        #    indices = indices[:, :, 1:]

        if input_pts.is_cuda:
            indices = indices.cuda()
            queries = queries.cuda()

        return indices, queries

    def indices_conv(self, input_pts, K):
        indices = nearest_neighbors.knn_batch(input_pts.cpu().detach().numpy(), input_pts.cpu().detach().numpy(), K, omp=True)
        indices = torch.from_numpy(indices).long()
        if input_pts.is_cuda:
            indices = indices.cuda()
        return indices, input_pts

    def indices_deconv(self, input_pts, next_pts, K):
        indices = nearest_neighbors.knn_batch(input_pts.cpu().detach().numpy(), next_pts.cpu().detach().numpy(), K, omp=True)
        indices = torch.from_numpy(indices).long()
        if input_pts.is_cuda:
            indices = indices.cuda()
        return indices, next_pts

    def points_processing(self, input, points, K, next_pts=None, normalize=True, indices_=None, return_indices=False, dilation=1):

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
            indices = indices[:, :, torch.randperm(indices.size(2))]
            indices = indices[:, :, :K]

        # compute indices for indexing points
        # if clouds are 16 with 1024 pts each , add_indices would be 1*1024, 2*1024, 3*1024, ... 16*1024
        # indices now will have all indices from 1 to 16*1024
        # the batch will run "in parallel" during the same forward() execution
        add_indices = torch.arange(batch_size).type(indices.type()) * n_pts
        # it is necessary to arrange add_indices to be compatible with indices dimension
        # note that size -1 is inferred from other dimensions
        # indices becomes a vector of ( NUMBER OF POINT CLOUDS  x NUMBER OF pts x NUMBER OF NEIGHBOURS K  )
        indices = indices + add_indices.view(-1, 1, 1)

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
        # pts being (BATCH_SIZE x POINT_PER_CLOUD x NEIGHBOUR x SPATIAL_DIM )
        next_pts_unsqueezed = next_pts.unsqueeze(2)
        pts = pts - next_pts_unsqueezed # was next_pts_unsqueezed

        # normalize to unit ball, or not
        if normalize:
            maxi = torch.sqrt((pts.detach() ** 2).sum(3).max(2)[0] )  # 1 is needed to put the maximum point inside the border of our sphere (instead of ON the border) detach is a modificaiton
            maxi[maxi == 0] = 1
            pts = pts / maxi.view(maxi.size() + (1, 1,))

        #move a bit the neighbourhood center
        # pts = pts.view(-1, 3)

        return pts, features, next_pts, indices_