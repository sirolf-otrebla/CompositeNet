from examples.deep_svdd.Trainer import *
# import pptk # for vizualizing point clouds 
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

#import imageio
from scipy import misc
from scipy.interpolate import interp1d
# import pyautogui, sys
from sklearn import metrics
import time

cfg_pool = [

    {

        # PARAMETERS FOR FOURIER-TRAINED RBFs
        #########################################

        # "pulse": 6.28,
        # "truncation": 8,

        # COMPOSITE LAYER PARAMETERS
        #########################################

        "n_centers": 56,                                                #  number of centers inside the spatial function
        "spatial_function_dimension": 14,                               # spatial function's output dimension
        "neighbours": 32,                                               # cardinality of each neighbourhood
        "spatial": "RBFN-norelu",                                       # kind of spatial function used. you can find some already implemented
        "semantic": "aggregate",                                        # kind of semantic function used. You can choose between aggregate or linear (convolutional)

        # ARCHITECTURE PARAMETERS
        #########################################

        "pl": 8,                                                        # called omega in the paper, decides the number of outgoing features from each network's layer
        "dropout": 0.5,
        "architecture": "CompositeNet",                                 # you can choose between CompositeNet and the original ConvPoint architecture
        "batchsize": 16,
        "npoints": 512,
        "biases": False,                                                # remove biases through the network

        # DEEP SVDD PARAMETERS
        #########################################

        "R": .1,                                                        # used only in soft-bound SVDD as in the paper by Ruff et al. ignore it if using the One-Class loss
        "c": 1000,                                                      # currently unused, since the SVDD center is set by computing the network's forward pass
        "nu": .6,                                                       # used only in soft-bound SVDD as in the paper by Ruff et al. ignore it if using the One-Class loss
        "center_fixed": True,                                           # the trainer does not update the center's position
        "soft_bound": False,                                            # Choose between One-Class or Soft-Bound Deep SVDD. In the paper, we employed One-Class Deep SVDD
        "output_dimension": 64,  # 128,                                 # dimension of the Deep SVDD output sphere
        "warm_up_n_epochs": 1,                                         # in the first epochs, the network is not tested. If using soft-bound loss, the radius is not updated.
        "noise_reg": True,                                              # adds random noise to the loss in order to prevent mode collapse

        # EXPERIMENT PARAMETERS
    #########################################

        "rootdir": "./data/shapenet",                                   # dataset's directory
        "savedir": "./exp_Aggregate_outDim64_Noise_hardLoss",           # directory where you want to save the output of the experiment
        "classes": [0,5,8,13,14,18,31,33,45,48,50],                     # classes used as training set
        "anomalies" : [1,2,3],                                          # classes  used as Anomalies. if None, all non_normal classes are used
        "repetitions" : 10,                                             # how many runs for each class
        "epoch_nbr": 0,                                                 # training epochs
        "ntree" : 1,
        "cuda" : True,                                                  # use Cuda or not


    # OTHER PARAMETERS
    #########################################

        "notes": " con CompositeLayer",

    },
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep SVDD network for CompositeNet. You can load an external configuration or use the example one already in the code')
    parser.add_argument('configs', metavar='C', nargs='+', default=None,
                        help='json path to desired network configuration. you can add more than one configuration and run them one after the other')
    args = parser.parse_args()
    if args.configs != None:
        cfg_pool = [ json.load(file) for file in [ open(path) for path in args.configs]]

    for c in cfg_pool:

        if c['cuda']:
            torch.backends.cudnn.benchmark = True
        
        for normal_class in c["classes"]:

            rocs = []
            aucs = []
            base_fpr = np.linspace(0, 1, 101)
            folderName = os.path.join(c["savedir"], str(normal_class))

            for i in range(0,c["repetitions"]):

                dataset = ADModelNetDataContainer(c["rootdir"], [normal_class], c["anomalies"])
                netFactory = modelBuilder(1, c["output_dimension"])  # 200
                c["class"] = normal_class
                net = netFactory.generate(c["architecture"], c)

                trainer = Trainer(
                    dataContainer=dataset, 
                    net=net, 
                    config=c, 
                    folderName=folderName)

                trainer.train(epoch_nbr=c["epoch_nbr"])

                np_expAuc = np.array(trainer.aucs)
                idx_max = np.argmax(np_expAuc)
                aucs.append(np_expAuc[idx_max])

                best_roc = trainer.rocs[idx_max]
                interp_roc = np.interp(base_fpr, best_roc[0], best_roc[1])
                rocs.append(interp_roc)

            # for each class, plot an average ROC curve
            aucs = np.array(aucs)
            mean_roc =  np.mean(rocs, axis=0)
            std_roc = np.std(rocs, axis=0)
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)

            plt.figure(figsize=(12, 8))
            plt.plot(base_fpr, mean_roc, 'b', alpha=0.8, label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc), )
            plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=0.8)
            plt.xlim([-0.01, 1.01])
            plt.ylim([-0.01, 1.01])
            plt.ylabel('True Positive Rate', fontsize = 24)
            plt.xlabel('False Positive Rate', fontsize = 24)
            plt.legend(loc="lower right",  prop={'size': 14})
            plt.title('Receiver operating characteristic (ROC) curve')
            # plt.axes().set_aspect('equal', 'datalim')
            plt.savefig(os.path.join(folderName, str(normal_class)+"_roc.png"))





