import os
import h5py

rootdir = "./data/shapenetcorev2_hdf5_2048"
mode = "train"
files = "train_files.txt"
N_CLASSES = 55
for cls in range(55):
        normal_classes = [cls]
        filenames = []
        for line in open(os.path.join(rootdir, files), "r"):
            line = line.split("\n")[0]
            line = os.path.basename(line)
            filenames.append(os.path.join(rootdir, line))

        data = []
        labels = []
        normal = []  # # furniture
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
            print("class \t %s \t has \t %d \t training samples" % (cls, len(labels)))
