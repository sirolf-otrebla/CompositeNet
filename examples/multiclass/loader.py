from examples.multiclass.Trainer import *
from examples.multiclass.dataContainer import *
config = [
    {

        # PARAMETERS FOR FOURIER-TRAINED RBFs
        #########################################

        # "pulse": 6.28,
        # "truncation": 8,

        # COMPOSITE LAYER PARAMETERS
        #########################################

        "n_centers": 56,  # 56,  # 48
        "spatial_function_dimension": 16,  # 14 # 6
        "neighbours": 32,
        "spatial": "RBFN-norelu",
        "semantic": "aggregate", # choices: aggregate or linear

        # ARCHITECTURE PARAMETERS
        #########################################

        "pl": 64,  #omega
        "dropout": 0.33,
        "architecture": "CompositeNet",
        "TL_path": None,  # "./save/AD_TL/state_dict.pth",
        "batchsize": 16,
        "npoints": 1024,
        "biases": False,

        # EXPERIMENT PARAMETERS
        #########################################

        "rootdir": "./data/multiclass",
        "savedir": "./saved_reults/",
        "epochs": 200,
        "ntree": 1,
        "cuda": True,
        "test": False,
        "schedule": [30, 60, 90],  # [20, 35, 50, 70],

        # OTHER PARAMETERS
        #########################################

        "notes": " ADD YOUR NOTES HERE",

    },
]

if __name__ == '__main__':

    for c in config:
        dataset = ModelNetDataContainer(c["rootdir"])
        netFactory = modelBuilder(1, len(dataset.getLabels()))
        net = netFactory.generate(c["architecture"], c)

        trainer = Trainer(dataContainer=dataset, net=net, config=c)

        trainer.train(epoch_nbr=c["epochs"])


