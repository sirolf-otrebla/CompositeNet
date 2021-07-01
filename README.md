# Composite Layers for Deep Anomaly detection on Point Clouds


## Introduction

This repository contains all the code writed during the development of #BMVC submission 1368.
The code is written in python, and makes use of the PyTorch library. Some components were originally
developed by A. Boulch for *ConvPoint*, and are available here:

https://github.com/aboulch/ConvPoint


## Platform and dependencies

The code was tested on Manjaro GNU/Linux "Nibia" 20.2.1, using a Conda environment with the following packages installed

- CUDAtoolkit 10.2.89
- CUDNN 7.6.5
- Cython 0.29.21
- Pytorch 1.9.0
- Scikit-learn 0.24.2
- TQDM 
- PlyFile
- H5py

All these dependencies can be installed via `conda install <package>` when using a conda environment.

## Nearest neighbor module
We use the same NN module used in *ConvPoint*. 
The ```CompositeLayer/knn``` directory contains a very small wrapper for [NanoFLANN](https://github.com/jlblancoc/nanoflann) with OpenMP.
Before executing the code, it is necessary to compile the module by running:
```
cd CompositeLayer/knn
python setup.py install --home="."
```

In the case, you do not want to use this C++/Python wrapper. You still can use the previous version of the nearest neighbors computation with Scikit Learn and Multiprocessing, python only version (slower). To do so, add the following lines at the start of your main script (e.g. ```modelnet_classif.py```):
```
from global_tags import GlobalTags
GlobalTags.legacy_layer_base(True)
```


## Datasets

We employ the already-sampled distributions of *ModelNet40* and *ShapeNetCore* available here:

https://github.com/AnTao97/PointCloudDatasets

Simply download and unzip them in the `./data` folder.

## Examples

### MultiClass Classification

you can run a multiclass classification by executing `./examples/multiclass/loader.py`. For your convenience,
you can also run the experiment by executing `launch_MC.sh` inside the main folder.

the experiment's configuration is contained inside a dictionary object at the start of `./examples/multiclass/loader.py`.
You can find a description of every configurable parameter in the comments.

You can add multiple configurations to `loader.py`, in order to execute several experiments in a row.

### Anomaly detection

you can run an anomaly detection experiment by executing `./examples/adetection/loader.py`. For your convenience,
you can also run the experiment by executing `launch_AD.sh` inside the main folder.

the experiment's configuration is contained inside a dictionary object at the start of `./examples/adetection/loader.py`.
You can find a description of every configurable parameter in the comments.

You can add multiple configurations to `loader.py`, in order to execute several experiments in a row.

