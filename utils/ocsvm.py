import os
from utils.svdd import SVDD
from sklearn.ensemble import IsolationForest

import utils.metrics
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm



GOOD_PATH = "../data/shapenet_GOOD_4"

TRAIN_GOOD = os.path.join(GOOD_PATH, "train")
TEST_GOOD = os.path.join(GOOD_PATH, "test")

kernelList = {"1": {"type": 'gauss', "width": 1e-4},
              "2": {"type": 'linear', "offset": 0},
              "3": {"type": 'ploy', "degree": 2, "offset": 0},
              "4": {"type": 'tanh', "gamma": 1e-4, "offset": 0},
              "5": {"type": 'lapl', "width": 1/12}
              }

parameters = {"positive penalty": 0.9,
              "negative penalty": 0.9,
              "kernel": kernelList.get("1"),
              "option": {"display": 'off'}}

normal_clss = [31,33]
repetitions = 10

if __name__ == '__main__':
    for i in normal_clss:
        auc_list = []
        roc_list = []
        for j in range(0,repetitions):
            NORMAL_CLASS = [i]
            train_data = []
            for cls in NORMAL_CLASS:
                read_dir = os.path.join(TRAIN_GOOD, str(cls))
                for pc in os.listdir(read_dir):
                    input_pc = os.path.join(read_dir, pc)
                    csv = np.genfromtxt(input_pc, delimiter=',')
                    csv = csv[:-1]
                    train_data.append(csv)

            test_data = []
            test_label = []
            ANOMALIES = [x for x in range(0,55)] #31 microphones
            ANOMALIES.remove(NORMAL_CLASS[0])
            for cls in NORMAL_CLASS + ANOMALIES:
                read_dir = os.path.join(TEST_GOOD, str(cls))
                for pc in os.listdir(read_dir):
                    input_pc = os.path.join(read_dir, pc)
                    csv = np.genfromtxt(input_pc, delimiter=',')
                    csv = csv[:-1]
                    test_data.append(csv)
                    if cls in NORMAL_CLASS:
                        test_label.append(1)   # NO in ifor and svm targets need to be the opposite
                    else:
                        test_label.append(-1)

            np_train_data = np.array(train_data)
            np.random.shuffle(np_train_data)
            np_test_data = np.array(test_data)
            np_test_label = np.array(test_label)
            number_of_rows = np_test_data.shape[0]
            random_indices = np.random.choice(number_of_rows, size=1500, replace=False)
            np_test_data = np_test_data[random_indices, :]
            np_test_label = np_test_label[random_indices]

            #clf = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=1e-4)
            clf = svm.OneClassSVM(nu=0.01, kernel="poly", degree=4)
            #clf = svm.OneClassSVM(nu=0.01, kernel="linear")
            #clf = IsolationForest(contamination=0, n_estimators=100)
            #svdd = SVDD(parameters)
            #svdd.train(np_train_data, np.zeros((np_train_data.shape[0],1)))
            clf.fit(np_train_data)

            base_fpr = np.linspace(0, 1, 101)

            #scores, accuracy = svdd.test(np_test_data, np_test_label)
            #scores = -scores
            scores = clf.decision_function(np_test_data)
            AUC = roc_auc_score(np_test_label, scores)
            roc_fpr, roc_tpr, roc_thr = roc_curve(np_test_label, scores)

            roc = np.interp(base_fpr, roc_fpr, roc_tpr)
            roc_list.append(roc)
            auc_list.append(AUC)

        mean_roc = np.mean(roc_list, axis=0)
        mean_auc = np.mean(auc_list, axis=0)
        std_auc = np.std(auc_list, axis=0)
        plt.figure(figsize=(12, 8))
        plt.plot(np.linspace(0, 1, 101), mean_roc, 'b', alpha=0.8, label=r'Mean ROC (AUC = %0.4f +/- %0.4f)' % (mean_auc, std_auc) )
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=0.8)
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.ylabel('True Positive Rate', fontsize=24)
        plt.xlabel('False Positive Rate', fontsize=24)
        plt.legend(loc="lower right", prop={'size': 14})
        plt.title('ROC curve for class %d' % i)
        plt.savefig("./roc_svm_poly_"+str(i)+".png")
        #plt.show()
    # plt.axes().set_aspect('equal', 'datalim')
    # plt.savefig(os.path.join(folderName, str(normal_class) + "_roc.png"))