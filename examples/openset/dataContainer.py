import h5py
import torch
import numpy as np
import torch.utils.data
import os

class OpenSetModelNetDataContainer():

    def __init__(self, rootDirectory, known_class_list, unknown_class_list):
        # first label is airplane, we consider it to be normal. the others are anomalies
        self.known_class_list = known_class_list
        self.labels = range(len(self.known_class_list)+1) # we add one label for all the unknowns
        self.unknown_class_list = unknown_class_list

        print("Getting train files...")
        self.train_data, self.train_labels, self.train_normals = self.get_data(rootDirectory, "train_files.txt", "train")
        print("Getting test files...")
        self.test_data, self.test_labels, self.test_normals = self.get_data(rootDirectory, "test_files.txt", "test")
        print("done")

    def get_data(self, rootdir, files, mode="train"): #38

        filenames = []
        for line in open(os.path.join(rootdir, files), "r"):
            line = line.split("\n")[0]
            line = os.path.basename(line)
            filenames.append(os.path.join(rootdir, line))

        data = []
        labels = []
        normal = []

        # test phase, only knowns
        if mode == "train":
            for filename in filenames:
                f = h5py.File(filename, 'r')
                keys = f.keys()
                for i in range(0, f["label"].shape[0]):
                    if f["label"][i] not in self.known_class_list:  # not normal class :) (chair)
                        pass
                    else:
                        labels.append(self.known_class_list.index(f["label"][i]))
                        data.append(f["data"][i, :])
                        if "normal" in keys:
                            normal.append(f["normal"][i, :])

        # test phase, adding also unknowns
        else:
            for filename in filenames:
                f = h5py.File(filename, 'r')
                keys = f.keys()
                for i in range(0, f["label"].shape[0]):
                    if f["label"][i] in self.known_class_list:
                        data.append(f["data"][i, :])
                        labels.append(self.known_class_list.index(f["label"][i]))
                        if "normal" in keys:
                            normal.append(f["normal"][i, :])
                    elif f["label"][i] in self.unknown_class_list:
                        data.append(f["data"][i, :])
                        labels.append(len(self.known_class_list)) # we add a new class representing unknowns
                        if "normal" in keys:
                            normal.append(f["normal"][i, :])
                    else:
                        pass

        labels = np.array(labels)
        data = np.array(data)
        if not normal:
            normals = None
        else:
            normals = np.array(normal)

        return data, labels, normals

    def getLabels(self):
        return self.labels

    def getTestData(self):
        return self.test_data

    def getTrainData(self):
        return self.train_data

    def getTrainLabels(self):
        return self.train_labels

    def getTestLabels(self):
        return self.test_labels

    def getDataLoader(self, numPts, threads, iterPerShape, batchSize):

        print("Creating dataloaders...", end="")
        ds = PointCloudFileLists(self.train_data, self.train_labels, pt_nbr=numPts) #normals removed
        train_loader = torch.utils.data.DataLoader(ds, batch_size=batchSize, shuffle=True, num_workers=threads)
        ds_test = PointCloudFileLists(self.test_data, self.test_labels, pt_nbr=numPts,
                                      training=False,
                                      num_iter_per_shape=iterPerShape) # same here
        test_loader = torch.utils.data.DataLoader(ds_test, batch_size=batchSize, shuffle=False,
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
