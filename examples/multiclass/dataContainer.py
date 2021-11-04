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
