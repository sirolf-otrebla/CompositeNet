import numpy as np
import random
import os
from tqdm import tqdm
import argparse
from datetime import datetime
from sklearn.metrics import confusion_matrix
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import utils.metrics as metrics


class SelfSupervisedDataContainer():

    def __init__(self, rootDirectory, normal_class_list, anomalies):
        # first label is airplane, we consider it to be normal. the others are anomalies

        self.labels = [
            "normal", "anomaly"
        ]

        self.SS_rotations = [

            create_3D_rotations(0),
            create_3D_rotations(45),
            create_3D_rotations(90),
            create_3D_rotations(135),
            create_3D_rotations(180),
            create_3D_rotations(225),
            create_3D_rotations(270),
            create_3D_rotations(315),
        ]

        self.normal_class_list = normal_class_list

        print("Getting train files...")
        self.train_data, self.train_labels, self.train_normals = self.get_data(rootDirectory, "train_files.txt", "train", anomalies)
        print("Getting test files...")
        self.test_data, self.test_labels, self.test_normals = self.get_data(rootDirectory, "test_files.txt", "test", anomalies)
        print("done")

    def get_data(self, rootdir, files, mode="train", anomalies=None): #38

        filenames = []
        for line in open(os.path.join(rootdir, files), "r"):
            line = line.split("\n")[0]
            line = os.path.basename(line)
            filenames.append(os.path.join(rootdir, line))

        data = []
        labels = []
        normal = []
        normal_classes = self.normal_class_list  # # furniture
        if mode == "train":
            for filename in filenames:
                f = h5py.File(filename, 'r')
                keys = f.keys()
                for i in range(0, f["label"].shape[0]):
                    if f["label"][i] not in normal_classes:  # not normal class :) (chair)
                        pass
                    else:
                        labels.append(0)
                        data.append(f["data"][i, :])
                        if "normal" in keys:
                            normal.append(f["normal"][i, :])
        else:
            for filename in filenames:
                f = h5py.File(filename, 'r')
                keys = f.keys()
                for i in range(0, f["label"].shape[0]):
                    if anomalies == None:
                        data.append(f["data"][i, :])
                        if "normal" in keys:
                            normal.append(f["normal"][i, :])
                        if f["label"][i] not in normal_classes:
                            labels.append(1)
                        else:
                            labels.append(0)
                    else:
                        if f["label"][i] not in normal_classes and f["label"][i] not in anomalies:  # not normal class :)
                            pass
                        else:
                            data.append(f["data"][i, :])
                            if "normal" in keys:
                                normal.append(f["normal"][i, :])
                            if f["label"][i] not in normal_classes:
                                labels.append(1)
                            else:
                                labels.append(0)

        labels = np.array(labels)
        data = np.array(data)
        if not normal:
            normals = None
        else:
            normals = np.array(normal)

        return data, labels, normals

    def getTransformationList(self):

        return [i for i in range(len(self.SS_rotations))]

    def getTestData(self):
        return self.test_data

    def getTrainData(self):
        return self.train_data

    def getTrainLabels(self):
        return self.train_labels

    def getTestLabels(self):
        return self.test_labels

    def getDataLoader(self, numPts, threads, batchSize):

        print("Creating dataloaders...", end="")
        ds = PointCloudFileLists(self.train_data, self.train_labels, self.SS_rotations, self.train_normals, pt_nbr=numPts)
        train_loader = torch.utils.data.DataLoader(ds, batch_size=batchSize*len(self.SS_rotations), shuffle=True, num_workers=threads)
        ds_test = PointCloudFileLists(self.test_data, self.test_labels, self.SS_rotations, self.test_normals, pt_nbr=numPts,
                                      training=False)
        test_loader = torch.utils.data.DataLoader(ds_test, batch_size=batchSize*len(self.SS_rotations), shuffle=False,
                                                  num_workers=threads)
        print("done")
        return train_loader, test_loader


class PointCloudFileLists(torch.utils.data.Dataset):
    """Main Class for Image Folder loader."""

    def __init__(self, data, labels, t_list, normals=None, pt_nbr=1024, training=True):
        """Init function."""

        self.SS_rotations = t_list
        self.data = data
        self.labels = labels
        self.training = training
        self.normals = normals
        self.pt_nbr = pt_nbr


        self.num_iter_per_shape = len(self.SS_rotations)

        # TODO put these inside general configuration
        class AugmentationConfig:

            def __init__(self):
                self.augment_rotation = 'vertical'
                self.augment_scale_min = 0.8
                self.augment_scale_max = 1.2
                self.augment_scale_anisotropic = True
                self.augment_symmetries = [False, False, False]
                self.augment_noise = 1e-4

        self.config = AugmentationConfig()


    def __getitem__(self, index):
        """Get item."""

        index_ = index // self.num_iter_per_shape

        # get the filename
        pts = self.data[index_]
        target = self.labels[index_]

        indices = np.random.choice(pts.shape[0], self.pt_nbr)
        pts = pts[indices]

        if self.normals != None:
            pts_normals = self.normals[index_]
            features = pts_normals[indices]
            pts, features, _, _2 = self.augmentation_transform(pts, features)

        else:
            features = np.ones((pts.shape[0], 1))
            pts, _, _2 = self.augmentation_transform(pts)

        # create features if normals are not used
        # features = np.ones((pts.shape[0], 1))

        pts = self.pc_normalize(pts)

        rotation_label = index % self.num_iter_per_shape

        R = self.SS_rotations[ rotation_label ]
        rotated_pts = np.sum(np.expand_dims(pts, 2) * R, axis=1)

        return rotated_pts.astype(np.float32), features.astype(np.float32), int(target), rotation_label, index_

    def __len__(self):
        """Length."""
        return self.data.shape[0] * self.num_iter_per_shape

    def pc_normalize(self, pc):
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    ################################
    # augmentation method from KPConv    
    ################################

    def augmentation_transform(self, points, normals=None, verbose=False):
        """Implementation of an augmentation transform for point clouds."""

        ##########
        # Rotation
        ##########

        # Initialize rotation matrix
        R = np.eye(points.shape[1])

        if points.shape[1] == 3:
            if self.config.augment_rotation == 'vertical':

                # Create random rotations
                theta = np.random.rand() * 2 * np.pi
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

            #elif self.config.augment_rotation == 'all':

                # Choose two random angles for the first vector in polar coordinates
                #theta = np.random.rand() * 2 * np.pi
                #phi = (np.random.rand() - 0.5) * np.pi

                # Create the first vector in carthesian coordinates
                #u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

                # Choose a random rotation angle
                #alpha = np.random.rand() * 2 * np.pi

                # Create the rotation matrix with this vector and angle
                #R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]


        R = R.astype(np.float32)

        #######
        # Scale
        #######

        # Choose random scales for each example
        min_s = self.config.augment_scale_min
        max_s = self.config.augment_scale_max
        if self.config.augment_scale_anisotropic:
            scale = np.random.rand(points.shape[1]) * (max_s - min_s) + min_s
        else:
            scale = np.random.rand() * (max_s - min_s) - min_s

        # Add random symmetries to the scale factor
        symmetries = np.array(self.config.augment_symmetries).astype(np.int32)
        symmetries *= np.random.randint(2, size=points.shape[1])
        scale = (scale * (1 - symmetries * 2)).astype(np.float32)

        #######
        # Noise
        #######

        noise = (np.random.randn(points.shape[0], points.shape[1]) * self.config.augment_noise).astype(np.float32)

        ##################
        # Apply transforms
        ##################

        # Do not use np.dot because it is multi-threaded
        # augmented_points = np.dot(points, R) * scale + noise
        augmented_points = np.sum(np.expand_dims(points, 2) * R, axis=1) * scale + noise

        if normals is None:
            return augmented_points, scale, R
        else:
            # Anisotropic scale of the normals thanks to cross product formula
            normal_scale = scale[[1, 2, 0]] * scale[[2, 0, 1]]
            augmented_normals = np.dot(normals, R) * normal_scale
            # Renormalise
            augmented_normals *= 1 / (np.linalg.norm(augmented_normals, axis=1, keepdims=True) + 1e-6)

            return augmented_points, augmented_normals, scale, R


def create_3D_rotations(theta):
    """
    Create rotation matrices from a list of axes and angles. Code from wikipedia on quaternions
    :param axis: float32[N, 3]
    :param angle: float32[N,]
    :return: float32[N, 3, 3]
    """
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [1, 0, 0]], dtype=np.float32)
    return R
"""
old

def create_3D_rotations(axis, angle):

    #Create rotation matrices from a list of axes and angles. Code from wikipedia on quaternions
    #:param axis: float32[N, 3]
    #:param angle: float32[N,]
    #:return: float32[N, 3, 3]

    t1 = np.cos(angle)
    t2 = 1 - t1
    t3 = axis[:, 0] * axis[:, 0]
    t6 = t2 * axis[:, 0]
    t7 = t6 * axis[:, 1]
    t8 = np.sin(angle)
    t9 = t8 * axis[:, 2]
    t11 = t6 * axis[:, 2]
    t12 = t8 * axis[:, 1]
    t15 = axis[:, 1] * axis[:, 1]
    t19 = t2 * axis[:, 1] * axis[:, 2]
    t20 = t8 * axis[:, 0]
    t24 = axis[:, 2] * axis[:, 2]
    R = np.stack([t1 + t2 * t3,
                  t7 - t9,
                  t11 + t12,
                  t7 + t9,
                  t1 + t2 * t15,
                  t19 - t20,
                  t11 - t12,
                  t19 + t20,
                  t1 + t2 * t24], axis=1)

    return np.reshape(R, (-1, 3, 3))



"""