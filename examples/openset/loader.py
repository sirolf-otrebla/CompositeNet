from examples.openset.Trainer import *
from examples.openset.dataContainer import *
from examples.openset.openmax import *
config = [
    {

        # PARAMETERS FOR FOURIER-TRAINED RBFs
        #########################################

        # "pulse": 6.28,
        # "truncation": 8,

        # COMPOSITE LAYER PARAMETERS
        #########################################

        "n_centers": 56,                                                 #  number of centers inside the spatial function
        "spatial_function_dimension": 16,                                # spatial function's output dimension
        "neighbours": 32,                                                # cardinality of each neighbourhood
        "spatial": "RBFN-norelu",                                        # kind of spatial function used. you can find some already implemented
        "semantic": "aggregate",                                         # kind of semantic function used. You can choose between aggregate or linear (convolutional)

        # ARCHITECTURE PARAMETERS
        #########################################

        "pl": 64,                                                        # called omega in the paper, decides the number of outgoing features from each network's layer
        "dropout": 0.33,                                                 # you can choose between CompositeNet and the original ConvPoint architecture
        "architecture": "CompositeNet",
        "TL_path": None,  # "./save/AD_TL/state_dict.pth",
        "batchsize": 16,
        "npoints": 1024,
        "biases": True,                                                  # remove biases through the network

        # EXPERIMENT PARAMETERS
        #########################################

        "rootdir": "./data/modelnet40_hdf5_2048",                        # dataset's directory
        "savedir": "./saved_reults/",                                    # directory where you want to save the output of the experiment
                                                                         # if testing, this directory has to contain a
                                                                         # network state named "state_dict.pth"
        "epochs": 50,
        "ntree": 1,
        "cuda": True,
        "test": False,
        "schedule": [30, 60, 90],                                        # learning rate schedule

        # OTHER PARAMETERS
        #########################################
        "known_classes" : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "unknown_classes" : [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        "notes": " ADD YOUR NOTES HERE",

    },
]



if __name__ == '__main__':

    for c in config:
        dataset = OpenSetModelNetDataContainer(c["rootdir"], c["known_classes"], c["unknown_classes"])
        netFactory = modelBuilder(1, 10)
        net = netFactory.generate(c["architecture"], c)
        trainer = Trainer(dataContainer=dataset, net=net, config=c)
        if c['test']:
            save_dir = c['savedir']
            net.load_state_dict(torch.load(os.path.join(save_dir, "state_dict.pth")))
            trainer.apply(0,training=False)
        else:
            trainer.train(epoch_nbr=c["epochs"])



