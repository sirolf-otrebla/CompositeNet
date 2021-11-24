import h5py
import torch
import numpy as np
import torch.utils.data
import os

class ModelNetDataContainer():


    def __init__(self, rootDirectory):
        self.labels = []
        for line in open(os.path.join(rootDirectory, "shape_names.txt"), "r"):
            line = line.split("\n")[0]
            self.labels.append(line)
        print("Getting train files...")
        self.train_data, self.train_labels = self.get_data(rootDirectory, "train_files.txt")
        print(self.train_data.shape, self.train_labels.shape)
        print("Getting test files...")
        self.test_data, self.test_labels = self.get_data(rootDirectory, "test_files.txt")
        print(self.test_data.shape, self.test_labels.shape)
        print("done")

    def get_data(self, rootdir, files):

        train_filenames = []
        for line in open(os.path.join(rootdir, files), "r"):
            line = line.split("\n")[0]
            line = os.path.basename(line)
            train_filenames.append(os.path.join(rootdir, line))

        data = []
        labels = []
        for filename in train_filenames:
            f = h5py.File(filename, 'r')
            data.append(f["data"])
            labels.append(f["label"][()].reshape(-1))

        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)

        return data, labels

    def getLabels(self):
        return self.labels

    def getTestData(self):
        return self.test_data

    def getTestLabels(self):
        return self.test_labels

    def getDataLoader(self, numPts, threads, iterPerShape, batchSize):
        print("Creating dataloaders...", end="")
        ds = PointCloudFileLists(self.train_data, self.train_labels, pt_nbr=numPts)
        train_loader = torch.utils.data.DataLoader(ds, batch_size=batchSize, shuffle=True, num_workers=threads)
        ds_test = PointCloudFileLists(self.test_data, self.test_labels, pt_nbr=numPts, training=False,
                                      num_iter_per_shape=iterPerShape)
        test_loader = torch.utils.data.DataLoader(ds_test, batch_size=batchSize, shuffle=True,
                                                  num_workers=threads)
        print("done")
        return train_loader, test_loader


class PointCloudFileLists(torch.utils.data.Dataset):
    """Main Class for Image Folder loader."""


    def __init__(self, data, labels, pt_nbr=1024, training=True, num_iter_per_shape=1):
        """Init function."""

        self.data = data
        self.labels = labels
        self.training = training
        self.pt_nbr = pt_nbr
        self.num_iter_per_shape = num_iter_per_shape

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

        # create features
        features = np.ones((pts.shape[0], 1))
        #pts, _, _2 = self.augmentation_transform(pts)

        pts = self.pc_normalize(pts)

        return pts.astype(np.float32), features.astype(np.float32), int(target), index_

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

            elif self.config.augment_rotation == 'all':

                # Choose two random angles for the first vector in polar coordinates
                theta = np.random.rand() * 2 * np.pi
                phi = (np.random.rand() - 0.5) * np.pi

                # Create the first vector in carthesian coordinates
                u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

                # Choose a random rotation angle
                alpha = np.random.rand() * 2 * np.pi

                # Create the rotation matrix with this vector and angle
                R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]

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


def create_3D_rotations(axis, angle):
    """
    Create rotation matrices from a list of axes and angles. Code from wikipedia on quaternions
    :param axis: float32[N, 3]
    :param angle: float32[N,]
    :return: float32[N, 3, 3]
    """

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

