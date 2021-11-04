import pyntcloud
import os
import pandas as pd
from examples.multiclass.dataContainer import ModelNetDataContainer


ROOT = "../data/shapenetcorev2_hdf5_2048"
NPOINTS=512
SAVE_PATH = "../data/shapenet_ply_512pts"

if __name__ == '__main__':

    dataset = ModelNetDataContainer(ROOT)
    train_data, train_labels = dataset.get_data(ROOT, "train_files.txt")
    test_data, test_labels = dataset.get_data(ROOT, "test_files.txt")

    for i in range(len(train_data)):

        points = pd.DataFrame(data = train_data[i,:], columns = ["x", "y", "z"])
        pc = pyntcloud.PyntCloud(points)
        pc = pc.get_sample("points_random", as_PyntCloud=True, n=NPOINTS)
        path = os.path.join(SAVE_PATH, "train")
        path = os.path.join(path, str(train_labels[i])) # [i,0]
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join(path, str(i)+".ply")
        pc.to_file(path)
        print("saved training PC n " + str(i) + "\n")

    for i in range(len(test_data)):

        points = pd.DataFrame(data = test_data[i,:], columns = ["x", "y", "z"])
        pc = pyntcloud.PyntCloud(points)
        pc = pc.get_sample("points_random", as_PyntCloud=True, n=NPOINTS)
        path = os.path.join(SAVE_PATH, "test")
        path = os.path.join(path, str(test_labels[i])) #[i,0]
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join(path, str(i)+".ply")
        pc.to_file(path)
        print("saved testing PC n " + str(i) + "\n")
