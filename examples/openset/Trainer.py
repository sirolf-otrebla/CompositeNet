import sys
sys.path.append('../../')

# other imports
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
import torch.utils.data
from matplotlib import pyplot as plt
import seaborn as sns
import utils.metrics as metrics
from examples.openset.dataContainer import *
from examples.openset.openmax import *
from examples.openset.evaluation import *


class modelBuilder():

    def __init__(self, input_channels, output_channels):
        self.input_channels = input_channels
        self.output_channels = output_channels

    def generate(self, model_name, config):
        print("Creating network")
        if model_name == "CompositeNet":
            from networks.network_classif import MCConpositeNet as Net
            return Net(self.input_channels, self.output_channels, config).float()

        else:
            from networks.network_classif import MCConvPoint as Net
            return Net(self.input_channels, self.output_channels).float()

class Trainer():

    def __init__(self, dataContainer, net, config, folderName=None):

        self.N_TRAIN_LABELS = len(dataContainer.getLabels())
        self.labels_list = dataContainer.getLabels()
        self.config = config
        config["n_parameters"] = self.count_parameters(net)
        # define the save directory
        if folderName == None:
            time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            folderName = "{}_bs{:02d}_pts{}_{}".format(config['architecture'], config['batchsize'], config['npoints'], time_string)
        self.save_dir = os.path.join(config['savedir'], folderName)
        # setting stuff for trainer
        if config['cuda']:
            net.cuda()
        print("Number of parameters", self.count_parameters(net))
        self.net = net
        self.train_loader, self.test_loader = dataContainer.getDataLoader(
            numPts=config['npoints'],
            threads=0,
            iterPerShape=1,
            batchSize=config['batchsize']
        )
        self.train_labels = dataContainer.getTrainLabels()
        self.test_data = dataContainer.getTestData()
        self.test_labels = dataContainer.getTestLabels()
        self.optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, config["schedule"], gamma=0.1) # gamma=0.5 [20, 35, 50, 70], gamma=0.5 ) [30, 45, 75]


    def init_logs(self):

        print(os.getcwd())
        os.makedirs(self.save_dir, exist_ok=True)
        logFile = open(os.path.join(self.save_dir, "logs.txt"), "w")
        configFile = open(os.path.join(self.save_dir, "config.txt"), "w")
        configFile.write(str(self.config))
        print("creating save folder")
        print(logFile)
        self.logFile = logFile
        return logFile


    def train(self, epoch_nbr=100):

        f = self.init_logs()
        for epoch in range(epoch_nbr):
            # TRAIN
            self.net.train()
            train_metrics = self.apply(epoch, training=True)
            # TEST
            self.net.eval()
            with torch.no_grad():
                test_metrics = self.apply(epoch, training=False)
            # UPDATE LEARNING RATE
            self.scheduler.step()
            # SAVE NETWORK
            torch.save(self.net.state_dict(), os.path.join(self.save_dir, "state_dict.pth"))
            # WRITE IN LOG FILE
            self.write_log_line(train_metrics.append(test_metrics))

        f.close()

    def apply(self, epoch, training=False):

        ''' Applies the function learnt by self.net over the input provided
    		by train and test loaders

    		Parameters:
    		-----------
    		epoch : int, used for logging purposes.
    		training : bool, if True we perform the backward pass,
    						 if False we compute test metrics.
    	'''
        error = 0
        cm = np.zeros((self.N_TRAIN_LABELS, self.N_TRAIN_LABELS))
        # ___________________________________________________________
        #
        # Training phase: weights are updated, returned metrics are
        #	only training_loss, training_OA, training_AA.
        # ___________________________________________________________
        #
        if training:
            t = tqdm(self.train_loader, desc="Epoch " + str(epoch), ncols=130)
            for pts, features, targets, indices in t:
                if self.config['cuda']:
                    features = features.cuda()
                    pts = pts.cuda()
                    targets = targets.cuda()
                self.optimizer.zero_grad()
                # FORWARD
                outputs = self.net(features, pts)
                targets = targets.view(-1)
                # BACKWARD STEP
                loss = F.cross_entropy(outputs, targets)
                loss.backward()
                self.optimizer.step()
                # METRICS IN TQDM PROGRESS BAR
                predicted_class = np.argmax(outputs.cpu().detach().numpy(), axis=1)
                target_np = targets.cpu().numpy()
                cm_ = confusion_matrix(target_np.ravel(), predicted_class.ravel(), labels=list(range(self.N_TRAIN_LABELS)))
                cm += cm_
                error += loss.item()
                oa = "{:.5f}".format(metrics.stats_overall_accuracy(cm))
                aa = "{:.5f}".format(metrics.stats_accuracy_per_class(cm)[0])
                aiou = "{:.5f}".format(metrics.stats_iou_per_class(cm)[0])
                aloss = "{:.5e}".format(error / cm.sum())
                t.set_postfix(OA=oa, AA=aa, AIOU=aiou, ALoss=aloss)

            return [aloss, oa, aa]
        # ___________________________________________________________
        #
        # Testing phase: weights are not updated, returned metrics are
        #	loss, OA, AA, AAUC, OAUC (see:
        # 	https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
        # ___________________________________________________________
        #
        else:

            predictions = np.zeros((self.test_data.shape[0], self.N_TRAIN_LABELS), dtype=float)
            t = tqdm(self.test_loader, desc="  Test " + str(epoch), ncols=100)
            for pts, features, targets, indices in t:
                if self.config['cuda']:
                    features = features.cuda()
                    pts = pts.cuda()
                    targets = targets.cuda()
                # FEEDING INPUT
                outputs = self.net(features, pts)
                targets = targets.view(-1)
                # COMPUTING LOSS FOR BATCH
                #loss = F.cross_entropy(outputs, targets)
                outputs_np = outputs.cpu().detach().numpy()
                for i in range(indices.size(0)):
                    predictions[indices[i]] += outputs_np[i]
                # UPDATING EPOCH ERROR

            # COMPUTING EPOCH METRICS
            scores = predictions[:, np.newaxis, :]

            pred_softmax, pred_softmax_threshold, pred_openmax, scores_softmax, scores_openmax = self.apply_openmax(scores)
            os_metrics  = self.compute_test_metrics(pred_openmax, scores_openmax, "Openmax")
            sts_metrics = self.compute_test_metrics(pred_openmax, scores_openmax, "Softmax_threshold")
            ss_metrics  = self.compute_test_metrics(pred_openmax, scores_openmax, "Softmax")

            return os_metrics.append(sts_metrics).append(ss_metrics)

    def count_parameters(self, model):
        parameters = model.parameters()
        return sum(p.numel() for p in parameters if p.requires_grad)


    def compute_test_metrics(self, predictions, scores, pred_name):
        eval = Evaluation(predictions, self.test_labels, scores)
        oa = "{:.5f}".format(eval.accuracy)
        aa = "{:.5f}".format(eval.f1_macro_weighted)
        aauc = "{:.5f}".format(eval.area_under_roc_ovr)
        oauc = "{:.5f}".format(eval.area_under_roc_ovo)
        print("Predictions for", pred_name, "OA", oa, "AA", aa, "AAUC", aauc, "OAUC", oauc)

        return [oa, aa, aauc, oauc]
    def apply_openmax(self, scores):

        print("\n Applying Openmax... \n")
        weibull_alpha = 3
        weibull_tail = 20
        weibull_threshold = 0.9
        _, mavs, dists = compute_train_score_and_mavs_and_dists(self.N_TRAIN_LABELS, self.train_loader, self.net)
        weibull_model = fit_weibull(mavs, dists, list(range(0, self.N_TRAIN_LABELS)), weibull_tail, "euclidean")


        pred_softmax, pred_softmax_threshold, pred_openmax = [], [], []
        score_softmax, score_openmax = [], []
        for score in scores:
            so, ss = openmax(weibull_model, list(range(0, self.N_TRAIN_LABELS)), score,
                             0.5, weibull_alpha, "euclidean")  # openmax_prob, softmax_prob
            pred_softmax.append(np.argmax(ss))
            pred_softmax_threshold.append(
                np.argmax(ss) if np.max(ss) >= weibull_threshold else self.N_TRAIN_LABELS)
            pred_openmax.append(np.argmax(so) if np.max(so) >= weibull_threshold else self.N_TRAIN_LABELS)
            score_softmax.append(ss)
            score_openmax.append(so)

        return pred_softmax, pred_softmax_threshold, pred_openmax, score_softmax, score_openmax

    def write_log_line(self, *args):
        for arg in args:
            self.logFile.write(arg)
            self.logFile.write(",")
        self.logFile.write("\n")


