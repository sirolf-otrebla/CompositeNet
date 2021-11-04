import sys

sys.path.append('../../')

# other imports

import torch.utils.data

from examples.self_supervised.dataContainer import *
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

        self.N_LABELS = len(dataContainer.getTransformationList())
        self.labels_list = dataContainer.getTransformationList()
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
            batchSize=config['batchsize']
        )

        self.rocs = []
        self.aucs = []

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

    def normality_score(self, O):
        return torch.mean(O, 1)


    def train(self, epoch_nbr=100):

        f = self.init_logs()
        for epoch in range(epoch_nbr):
            # TRAIN
            self.net.train()
            train_aloss, train_oa, train_aa = self.apply(epoch, training=True)
            # TEST
            self.net.eval()
            self.net.eval()
            with torch.no_grad():
                test_auc, roc, self.outputs = self.apply(epoch, training=False)
                self.rocs.append(roc)
                self.aucs.append(float(test_auc))
            # save network
            torch.save(self.net.state_dict(), os.path.join(self.save_dir, "state_dict.pth"))
            # write the logs
            f.write(str(epoch) + ",")
            f.write(train_aloss + ",")
            f.write(test_auc + "\n")
            f.flush()

        self.scheduler.step()
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
        cm = np.zeros((self.N_LABELS, self.N_LABELS))
        # ___________________________________________________________
        #
        # Training phase: weights are updated, returned metrics are
        #	only training_loss, training_OA, training_AA.
        # ___________________________________________________________
        #
        if training:
            t = tqdm(self.train_loader, desc="Epoch " + str(epoch), ncols=130)
            for pts, features, _ , target_transform, indices in t:
                if self.config['cuda']:
                    features = features.cuda()
                    target_transform = target_transform.cuda()
                    pts = pts.cuda()
                self.optimizer.zero_grad()
                # FORWARD
                outputs = self.net(features, pts)
                target_transform = target_transform.view(-1)
                # BACKWARD STEP
                loss = F.cross_entropy(outputs, target_transform)
                loss.backward()
                self.optimizer.step()
                # METRICS IN TQDM PROGRESS BAR
                predicted_class = np.argmax(outputs.cpu().detach().numpy(), axis=1)
                target_np = target_transform.cpu().numpy()
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
            anomaly_scores = np.zeros((self.test_data.shape[0], self.N_LABELS), dtype=float)
            t = tqdm(self.test_loader, desc="  Test " + str(epoch), ncols=100)
            for pts, features, targets, targets_transform, indices in t:
                if self.config['cuda']:
                    features = features.cuda()
                    # target_transform = targets_transform.cuda()
                    pts = pts.cuda()
                    # targets = targets.cuda()
                # FEEDING INPUT
                outputs = self.net(features, pts)
                # COMPUTE ANOMALY SCORE
                outputs = outputs.view(-1, self.N_LABELS)
                batch_scores = 1 - self.normality_score(outputs)

                batch_scores = batch_scores.cpu().detach().numpy()
                for i in range(indices.size(0)):
                    anomaly_scores[indices[i]] += batch_scores[i]

            auc = "{:.4f}".format(metrics.roc_auc_score(self.test_labels, anomaly_scores))
            print("Predictions", "AUC", auc)
            # "AA", aa,
            # "IOU", aiou,
            # "normAcc", normAcc,
            # "anomAcc", anomAcc)
            roc_fpr, roc_tpr, roc_thr = metrics.roc_curve(self.test_labels, anomaly_scores)
            return auc, [roc_fpr, roc_tpr, roc_thr], anomaly_scores

    def count_parameters(self, model):
        parameters = model.parameters()
        return sum(p.numel() for p in parameters if p.requires_grad)