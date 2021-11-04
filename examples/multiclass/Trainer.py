import sys
import time

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
from examples.multiclass.dataContainer import *
#from torch.profiler import profile, record_function, ProfilerActivity

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

        self.N_LABELS = len(dataContainer.getLabels())
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
        return logFile


    def train(self, epoch_nbr=100):

        f = self.init_logs()
        for epoch in range(epoch_nbr):
            # TRAIN
            self.net.train()
            train_aloss, train_oa, train_aa = self.apply(epoch, training=True)
            # TEST
            self.net.eval()
            with torch.no_grad():
                test_aloss, test_oa, test_aa,  test_aauc, test_oauc, cm = self.apply(epoch, training=False)
            # UPDATE LEARNING RATE
            self.scheduler.step()
            # SAVE NETWORK
            torch.save(self.net.state_dict(), os.path.join(self.save_dir, "state_dict.pth"))
            # WRITE IN LOG FILE
            f.write(str(epoch)+",")
            f.write(train_aloss+",")
            f.write(train_oa+",")
            f.write(train_aa+",")
            f.write(test_aloss+",")
            f.write(test_oa+",")
            f.write(test_aa+",")
            f.write(test_oauc+"\n")
            f.flush()

        f.close()

        # saving confusion matrix in text format and as heatmap image
        # ___________________________________________________________
        print(os.getcwd())
        os.makedirs(self.save_dir, exist_ok=True)
        np.set_printoptions(threshold=sys.maxsize)
        auc_file = open(os.path.join(self.save_dir, "confusion.txt"), "w")
        auc_file.write(str(cm))
        auc_file.close()
        plt.figure(figsize=(30, 20))
        cm_sum = cm.sum(axis=1)
        cm_norm = cm / cm_sum[:, np.newaxis]
        ax = sns.heatmap(cm_norm, fmt="d")
        ax.figure.savefig(os.path.join(self.save_dir, "confusion.png"))
        np.set_printoptions(threshold=1000)

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
        cm = np.zeros((self.N_LABELS, self.N_LABELS))
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
                cm_ = confusion_matrix(target_np.ravel(), predicted_class.ravel(), labels=list(range(self.N_LABELS)))
                cm += cm_
                error += loss.item()
                oa = "{:.5f}".format(metrics.stats_overall_accuracy(cm))
                aa = "{:.5f}".format(metrics.stats_accuracy_per_class(cm)[0])
                aiou = "{:.5f}".format(metrics.stats_iou_per_class(cm)[0])
                aloss = "{:.5e}".format(error / cm.sum())
                t.set_postfix(OA=oa, AA=aa, AIOU=aiou, ALoss=aloss)

            return aloss, oa, aa
        # ___________________________________________________________
        #
        # Testing phase: weights are not updated, returned metrics are
        #	loss, OA, AA, AAUC, OAUC (see:
        # 	https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
        # ___________________________________________________________
        #
        else:
            times_list = []
            predictions = np.zeros((self.test_data.shape[0], self.N_LABELS), dtype=float)
            t = tqdm(self.test_loader, desc="  Test " + str(epoch), ncols=100)
            for pts, features, targets, indices in t:
                if self.config['cuda']:
                    features = features.cuda()
                    pts = pts.cuda()
                    targets = targets.cuda()
                # FEEDING INPUT
                then = time.time()
                outputs = self.net(features, pts)
                elapsed = time.time() - then
                times_list.append(elapsed*1000)
                targets = targets.view(-1)
                # COMPUTING LOSS FOR BATCH
                loss = F.cross_entropy(outputs, targets)
                outputs_np = outputs.cpu().detach().numpy()
                for i in range(indices.size(0)):
                    predictions[indices[i]] += outputs_np[i]
                # UPDATING EPOCH ERROR
                error += loss.item()
                # Updating metrics in tqdm progress bar
                if self.config['ntree'] == 1:
                    pred_labels = np.argmax(outputs_np, axis=1)
                    targets_np = targets.cpu().numpy()
                    cm_ = confusion_matrix(targets.cpu().numpy(), pred_labels, labels=list(range(self.N_LABELS)))
                    cm += cm_
                    oa = "{:.5f}".format(metrics.stats_overall_accuracy(cm))
                    aa = "{:.5f}".format(metrics.stats_accuracy_per_class(cm)[0])
                    aiou = "{:.5f}".format(metrics.stats_iou_per_class(cm)[0])
                    aloss = "{:.5e}".format(error / cm.sum())
                    t.set_postfix(OA=oa, AA=aa, AIOU=aiou, ALoss=aloss)

            print("mean time for batch: {:.5f} ms".format(np.mean(times_list)))
            # COMPUTING EPOCH METRICS
            predicted_classes = np.argmax(predictions, axis=1)
            scores_np = predictions - np.expand_dims(np.min(predictions, axis=1), axis=1)
            scores_np = scores_np / np.expand_dims(np.max(scores_np, axis=1), axis=1)
            scores_np = scores_np / np.expand_dims(scores_np.sum(axis=1), axis=1)
            cm = confusion_matrix(self.test_labels, predicted_classes, labels=list(range(self.N_LABELS)))
            oa = "{:.5f}".format(metrics.stats_overall_accuracy(cm))
            aa = "{:.5f}".format(metrics.stats_accuracy_per_class(cm)[0])
            aloss = "{:.5e}".format(error / cm.sum())
            aauc = "{:.5f}".format(metrics.roc_auc_score(self.test_labels.squeeze(), scores_np, average="weighted", multi_class="ovo", labels=list(range(self.N_LABELS))))
            oauc = "{:.5f}".format(metrics.roc_auc_score(self.test_labels.squeeze(), scores_np, average="macro", multi_class="ovo", labels=list(range(self.N_LABELS))))
            print("Predictions", "loss", aloss, "OA", oa, "AA", aa, "AAUC", aauc, "OAUC", oauc)

            return aloss, oa, aa, aauc, oauc, cm

    def count_parameters(self, model):
        parameters = model.parameters()
        return sum(p.numel() for p in parameters if p.requires_grad)
