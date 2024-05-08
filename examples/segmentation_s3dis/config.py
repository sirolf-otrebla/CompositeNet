config = {

        # PARAMETERS FOR FOURIER-TRAINED RBFs
        #########################################

        # "pulse": 6.28,
        # "truncation": 8,

        # COMPOSITE LAYER PARAMETERS
        #########################################

        "n_centers": 128,                                                #  number of centers inside the spatial function
        "spatial_function_dimension": 16,                                # spatial function's output dimension
        "neighbours": 32,                                                # cardinality of each neighbourhood
        "spatial": "RBFN-norelu",                                        # kind of spatial function used. you can find some already implemented
        "semantic": "linear",                                         # kind of semantic function used. You can choose between aggregate or linear (convolutional)

        # ARCHITECTURE PARAMETERS
        #########################################

        "pl": 16,                                                        # called J_0 in the paper, decides the number of outgoing features from each network's layer
        "dropout": 0.0,                                                 # you can choose between CompositeNet and the original ConvPoint architecture
        "architecture": "CompositeNet",
        "TL_path": None,  # "./save/AD_TL/state_dict.pth",
        "batchsize": 2,
        "npoints": 2048,
        "biases": True,                                                  # remove biases through the network

        # EXPERIMENT PARAMETERS
        #########################################

                                                                         # if testing, this directory has to contain a
                                                                         # network state named "state_dict.pth"
        "epochs": 200,
        "ntree": 1,
        "cuda": True,
        "test": True,
        "schedule": [15,30,45],                                        # learning rate schedule

        # OTHER PARAMETERS
        #########################################

        "notes": " ADD YOUR NOTES HERE",
    }