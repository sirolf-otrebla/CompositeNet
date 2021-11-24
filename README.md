# Composite Layers for Deep Anomaly detection on Point Clouds


## Introduction

This repository contains all the code writed during the development of CVPR submission 8953.
The code is written in python, and makes use of the PyTorch library. Some components were originally
developed by A. Boulch for *ConvPoint*, and are available here:

https://github.com/aboulch/ConvPoint


## Platform and dependencies

The code was tested on Ubuntu GNU/Linux 18.04, using a Conda environment with the following packages installed:

- CUDAtoolkit 11.4
- CUDNN 7.6.5
- Cython 0.29.21
- Pytorch 1.9.0
- Scikit-learn 0.24.2
- TQDM 
- PlyFile
- H5py
- Matplotlib
- Seaborn

All these dependencies can be installed via `conda install <package>` when using a conda environment. The testing hardware configuration is the following:

- AMD Ryzen Threadripper 1950X 16-Core Processor
- NVIDIA RTX A6000 w/ 48GiB VRAM

The code was developed to be run over a CUDA GPU. Note that in some older video cards issues may arise during training.

It is possible, though not recommended, to execute
the code over the CPU by setting `cuda : False` inside each experiment's configuration. 

## Nearest neighbor module
We use the same NN module used in *ConvPoint*. 
The ```CompositeLayer/knn``` directory contains a very small wrapper for [NanoFLANN](https://github.com/jlblancoc/nanoflann) with OpenMP.
Before executing the code, it is necessary to compile the module by running:
```
cd CompositeLayer/knn
python setup.py install --home="."
```

You can also use the nearest neighbors computation with Scikit Learn and Multiprocessing, python only version (slower). To do so, add the following lines at the start of your main script (e.g. ```modelnet_classif.py```):
```
from global_tags import GlobalTags
GlobalTags.legacy_layer_base(True)
```


## Datasets

We employ the already-sampled distributions of *ModelNet40* and *ShapeNetCore* available here:

https://github.com/AnTao97/PointCloudDatasets

Simply download and unzip them in the `./data` folder.

Unfortunately, its licence prevents us or others to redistribute the *ScanNet* dataset. If you wan to run experiments on it, you will have to extract the shapes directly from the original segmentation dataset. You can find more here:'

http://www.scan-net.org/


## Examples

### MultiClass Classification

you can run a multiclass classification by executing `./examples/multiclass/loader.py`.

`python3 -m examples.multiclass.loader `

the experiment's configuration is contained inside a dictionary object at the start of `./examples/multiclass/loader.py`.
You can find a description of every configurable parameter in the comments.

You can load an external configuration or use the example one already in the code. To do that, simply write your configuration in a JSON file and load it by adding its path as CLI argument. For example:

`python3 -m examples.multiclass.loader myConfig.json ` 

You can also fed loader.py with more than one configuration at a time: they will be exectued sequentially.

` python3 -m examples.multiclass.loader ./myConfig1.json ./myConfig2.json  ./myConfig3.json` 

### Deep SVDD

you can run a Deep SVDD experiment by executing `./examples/deep_svdd/loader.py`.

`python3 -m examples.deep_svdd.loader `

the experiment's configuration is contained inside a dictionary object at the start of `./examples/deep_svdd/loader.py`.
You can find a description of every configurable parameter in the comments.

You can load an external configuration or use the example one already in the code. To do that, simply write your configuration in a JSON file and load it by adding its path as CLI argument. For example:

`python3 -m examples.deep_svdd.loader myConfig.json ` 

You can also fed loader.py with more than one configuration at a time: they will be exectued sequentially.

` python3 -m examples.deep_svdd.loader ./myConfig1.json ./myConfig2.json  ./myConfig3.json` 

### Self-Supervised AD

you can run a Self-Supervised AD experiment by executing `./examples/self_supervised/loader.py`. For your convenience,

`python3 -m examples.self_supervised.loader `

the experiment's configuration is contained inside a dictionary object at the start of `./examples/self_supervised/loader.py`.
You can find a description of every configurable parameter in the comments.

You can load an external configuration or use the example one already in the code. To do that, simply write your configuration in a JSON file and load it by adding its path as CLI argument. For example:

` python3 -m examples.self_supervised.loader myConfig.json ` 

You can also fed loader.py with more than one configuration at a time: they will be exectued sequentially.

` python3 -m examples.self_supervised.loadery ./myConfig1.json ./myConfig2.json  ./myConfig3.json` 

