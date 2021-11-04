from examples.multiclass.Trainer import *
from examples.multiclass.dataContainer import *
from torch import cuda, device
#from torch.profiler import profile, record_function, ProfilerActivity
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
        "semantic": "MLP",                                         # kind of semantic function used. You can choose between aggregate or linear (convolutional)

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

        "rootdir": "./data/scannet",                        # dataset's directory
        "savedir": "./saved_reults/MC_scannet_aggregate",                                    # directory where you want to save the output of the experiment
                                                                         # if testing, this directory has to contain a
                                                                         # network state named "state_dict.pth"
        "epochs": 100,
        "ntree": 1,
        "cuda": True,
        "test": False,
        "schedule": [30, 60, 90],                                        # learning rate schedule

        # OTHER PARAMETERS
        #########################################

        "notes": " ADD YOUR NOTES HERE",

    },
]


print(cuda.device_count())
for d in range(cuda.device_count()):
    if cuda.get_device_name(device(d)).startswith("Tesla"):
        cuda.set_device(d)

print(cuda.current_device(), cuda.get_device_name(device(d)))

if __name__ == '__main__':

    for c in config:
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

