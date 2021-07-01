import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from global_tags import GlobalTags
if GlobalTags.legacy_layer_base():
    from ..legacy.layer_base import LayerBase
else:
    from ..layer_base import LayerBase



class AggregateSemanticLayer(nn.Module):
    # 75 aa 81 OA
    def  __init__(self, input_features, output_features, config, use_bias=False, relu=False):

        super(AggregateSemanticLayer, self).__init__()

        self.spatial_function_dimension = config["spatial_function_dimension"]
        self.input_features = input_features
        self.output_features = output_features
        self.use_relu = relu

        # Weights
        self.feature_linear = nn.Linear(4*self.spatial_function_dimension*input_features, output_features, use_bias)
        weight = self.feature_linear.weight
        nn.init.kaiming_normal_(weight)
        self.relu = nn.LeakyReLU(inplace=True)
        self.tanh = nn.Tanh()

        self.bn = nn.BatchNorm1d(self.spatial_function_dimension)


    def forward(self, features, spatial_layer, K):

        BATCH_SIZE = spatial_layer.size(0)
        PTS_PER_POINT_CLOUD = spatial_layer.size(1)
        NEIGHBOURS = K

        fs = features.size()
        spatial_layer_var = spatial_layer.var(2)
        spatial_layer_mean = spatial_layer.mean(2)

        spatial_layer_var = spatial_layer_var.view(BATCH_SIZE, PTS_PER_POINT_CLOUD, 1, self.spatial_function_dimension)
        spatial_layer_mean = spatial_layer_mean.view(BATCH_SIZE, PTS_PER_POINT_CLOUD, 1, self.spatial_function_dimension)

        spatial_layer = torch.cat((spatial_layer_mean, spatial_layer_var), dim=3)

        feaures_mean = features.mean(2)
        features_std = features.std(2)
        features = torch.cat((feaures_mean, features_std), dim=2)
        features = features.view(BATCH_SIZE, PTS_PER_POINT_CLOUD, 2*self.input_features, 1)

        #features = features / fs[2]

        out = self.feature_linear((features @ spatial_layer).view(BATCH_SIZE, PTS_PER_POINT_CLOUD, -1))
        if self.use_relu:
            out = self.relu(out)

        return out

    def freeze(self):
        for p in self.feature_linear.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.feature_linear.parameters():
            p.requires_grad = True

    def reset(self):
        self.feature_linear = nn.Linear(self.spatial_function_dimension*self.input_features, self.output_features, self.use_bias)


class LinearSemanticLayer(nn.Module):

    def __init__(self, input_features, output_features, config, use_bias=True, relu=False):

        super(LinearSemanticLayer, self).__init__()

        self.spatial_function_dimension = config["spatial_function_dimension"]
        self.input_features = input_features
        self.output_features = output_features
        self.use_relu = relu
        self.use_bias = config["biases"]
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_features), requires_grad=True)
            self.bias.data.uniform_(-0.1,0.1)

        # Weights
        self.feature_weight = nn.Parameter(
                       torch.Tensor(input_features, self.spatial_function_dimension, output_features).float(), requires_grad=True)
        nn.init.kaiming_normal_(self.feature_weight.data)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()


    def forward(self, features, spatial_layer, K):


        fs = features.size()
        features = features.transpose(2,3)
        features = features.view(-1, features.size(2), features.size(3))
        spatial_layer = spatial_layer.view(-1, spatial_layer.size(2), spatial_layer.size(3))


        features = torch.bmm(features, spatial_layer)

        features = features.view(fs[0], fs[1], -1)

        features = torch.matmul(features, self.feature_weight.view(-1, self.feature_weight.size(2)))
        features = features/ fs[2]  #fs[0]

        if self.use_bias:
            features = features + self.bias

        return features

    def freeze(self):
        self.feature_weight.requires_grad = False
    def unfreeze(self):
        self.feature_weight.requires_grad = True

    def reset(self):
        nn.init.kaiming_normal_(self.feature_weight.data)
