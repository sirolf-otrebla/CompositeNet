from examples.adetection.Trainer import *
# import pptk # for vizualizing point clouds 
import numpy as np
import matplotlib.pyplot as plt

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

        #"pulse": 6.28,
        #"truncation": 8,
        
    # COMPOSITE LAYER PARAMETERS
    #########################################
    
        "n_centers": 56, #56,  # 48
        "spatial_function_dimension":14, #14 # 6
        "neighbours": 32,
        "spatial": "RBFN-norelu",
        "semantic": "aggregate",


    # ARCHITECTURE PARAMETERS
    #########################################

        "pl": 8, #8,  # 16,
        "dropout": 0.5,
        "architecture": "CompositeNet",
        "TL_path": None,  # "./save/AD_TL/state_dict.pth",
        "batchsize" : 16,
        "npoints" : 1024,
        "biases": False,

    # DEEP SVDD PARAMETERS
    #########################################

        "R": .1,
        "c": 1000,
        "nu": .6,
        "center_fixed": True,
        "soft_bound" : False,
        "output_dimension": 128,  # 128,
        "warm_up_n_epochs" : 15,
        "noise_reg" : True,
 
    # EXPERIMENT PARAMETERS
    #########################################

        "rootdir": "./data/shapenet",
        "savedir": "./exp_Aggregate_sameParameters_Noise_hardLoss",
        "classes": [0, 5, 8, 13, 14, 18, 31, 33, 45, 48, 50],                 # each class will be repeated $repetitions times
        "anomalies" : [1,2,3],
        "repetitions" : 10,
        "epoch_nbr": 20,
        "ntree" : 1,
        "cuda" : True,
        "test" : False,


    # OTHER PARAMETERS
    #########################################

        "notes": " con CompositeLayer",

    },
]

if __name__ == '__main__':

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





