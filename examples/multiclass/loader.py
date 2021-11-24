from examples.multiclass.Trainer import *
from examples.multiclass.dataContainer import *
from torch import cuda, device
import argparse
import json
#from torch.profiler import profile, record_function, ProfilerActivity
cfg_pool = [
    {

        # PARAMETERS FOR FOURIER-TRAINED RBFs
        #########################################

        # "pulse": 6.28,
        # "truncation": 8,

        # COMPOSITE LAYER PARAMETERS
        #########################################

        "n_centers": 256,                                                #  number of centers inside the spatial function
        "spatial_function_dimension": 16,                                # spatial function's output dimension
        "neighbours": 32,                                                # cardinality of each neighbourhood
        "spatial": "RBFN-norelu",                                        # kind of spatial function used. you can find some already implemented
        "semantic": "aggregate",                                         # kind of semantic function used. You can choose between aggregate or linear (convolutional)

        # ARCHITECTURE PARAMETERS
        #########################################

        "pl": 64,                                                        # called J_0 in the paper, decides the number of outgoing features from each network's layer
        "dropout": 0.33,                                                 # you can choose between CompositeNet and the original ConvPoint architecture
        "architecture": "ConvPoint",
        "TL_path": None,  # "./save/AD_TL/state_dict.pth",
        "batchsize": 16,
        "npoints": 1024,
        "biases": True,                                                  # remove biases through the network

        # EXPERIMENT PARAMETERS
        #########################################

        "rootdir": "./data/ShapeNetCore_hdf5_2048",                                # dataset's directory
        "savedir": "./saved_reults/myExp",                               # directory where you want to save the output of the experiment
                                                                         # if testing, this directory has to contain a
                                                                         # network state named "state_dict.pth"
        "epochs": 200,
        "ntree": 1,
        "cuda": True,
        "test": False,
        "schedule": [50,100,150],                                        # learning rate schedule

        # OTHER PARAMETERS
        #########################################

        "notes": " ADD YOUR NOTES HERE",

    },
]
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multiclass classification network for CompositeNet. You can load an external configuration or use the example one already in the code')
    parser.add_argument('configs', metavar='C', nargs='*', default=None,
                        help='json path to desired network configuration. you can add more than one configuration and run them one after the other')
    args = parser.parse_args()
    if args.configs != []:
        cfg_pool = [ json.load(file) for file in [ open(path) for path in args.configs]]

    for c in cfg_pool:
        dataset = ModelNetDataContainer(c["rootdir"])
        print("n_classes %d" % len(dataset.getLabels()))
        netFactory = modelBuilder(1, len(dataset.getLabels()))
        net = netFactory.generate(c["architecture"], c)
        trainer = Trainer(dataContainer=dataset, net=net, config=c)
        if c['test']:
            save_dir = c['savedir']
            net.load_state_dict(torch.load(os.path.join(save_dir, "state_dict.pth")))
            trainer.apply(0,training=False)
        else:
            trainer.train(epoch_nbr=c["epochs"])

