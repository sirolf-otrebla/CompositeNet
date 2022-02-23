from CompositeLayer.nn.Conv import LayerBase, CompositeConv
import CompositeLayer.nn.Conv
import torch
import torch.nn as nn
import torch.nn.functional as F

class Layer1x1( CompositeLayer.nn.Conv.LayerBase):

    def __init__(self, input_features, output_features, config, use_bias=True, relu=False):
        super(LayerBase, self).__init__()

        self.input_features = input_features
        self.output_features = output_features
        self.mlp = nn.Linear(self.input_features, output_features, bias=config["biases"])

        self.relu = nn.ReLU(inplace=True)


    def forward(self, input, points, next_pts=None, normalize=True, indices_=None, return_indices=False, dilation=1):

        pts, features, next_pts, indices_ = self.points_processing(input, points, 1, next_pts, normalize, indices_, return_indices, dilation)

        BATCH_SIZE = features.size(0)
        PTS_PER_POINT_CLOUD = features.size(1)

        features = features.view(BATCH_SIZE, PTS_PER_POINT_CLOUD, -1)
            # spatial_layer = spatial_layer.view(BATCH_SIZE, PTS_PER_POINT_CLOUD, NEIGHBOURS, -1)

        out = self.mlp(features)
        print(out.shape)
        if return_indices:
            return out, next_pts, indices_
        else:
            return out, next_pts


class AveragePool( CompositeLayer.nn.Conv.LayerBase):

    def __init__(self):
        super(LayerBase, self).__init__()

    def forward(self, input, points, K, next_pts=None, normalize=True, indices_=None, return_indices=False, dilation=1):

        pts, features, next_pts, indices_ = self.points_processing(input, points, K, next_pts, normalize, indices_,
                                                                   return_indices, dilation)

        BATCH_SIZE = features.size(0)
        PTS_PER_POINT_CLOUD = features.size(1)
        NEIGHBOURS = K

        features = features.view(BATCH_SIZE, PTS_PER_POINT_CLOUD, NEIGHBOURS, -1)
        # spatial_layer = spatial_layer.view(BATCH_SIZE, PTS_PER_POINT_CLOUD, NEIGHBOURS, -1)

        out = torch.mean(features, dim=2)

        if return_indices:
            return out, next_pts, indices_
        else:
            return out, next_pts



class ResnetBlock( CompositeLayer.nn.Conv.LayerBase):

    def __init__(self, input_features, output_features, config, dim, spatial_id, semantic_id ):

        super(LayerBase, self).__init__()

        self.pool = AveragePool()
        self.shortcut_1x1 = Layer1x1(input_features, output_features, config, use_bias=True, relu=False)
        self.pre_1x1 = Layer1x1(input_features, input_features//2, config, use_bias=True, relu=False)
        self.conv = CompositeConv(input_features//2, output_features//2, config, dim, spatial_id, semantic_id )
        self.post_1x1 = Layer1x1(output_features//2, output_features, config, use_bias=True, relu=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, points, K, next_pts=None):

        input_f, pts = self.pool(input, points, K, next_pts)

        main_f, _ = self.pre_1x1(input_f, pts)
        main_f =  self.relu(main_f)
        main_f, _ = self.conv(main_f, pts, K)
        main_f, _ = self.post_1x1(main_f, pts)
        main_f =  self.relu(main_f)

        shortcut_f, _ = self.relu(self.shortcut_1x1(input_f, pts))
        shortcut_f = self.relu(shortcut_f)

        out_f = main_f + shortcut_f

        return out_f, pts
