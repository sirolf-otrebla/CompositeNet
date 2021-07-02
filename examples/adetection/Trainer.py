import sys
sys.path.append('../../')

# other imports
import torch.utils.data
from examples.adetection.dataContainer import *

class modelBuilder():

    def __init__(self, input_channels, output_channels=100):
        self.input_channels = input_channels
        self.output_channels = output_channels

    def generate(self, model_name, config):
        print("Creating network")
        # loading our model
        if model_name == "CompositeNet":
            from networks.network_ad import ADCompositeNet as Net
        # loading CompositeLayer
        else:
            from networks.network_ad import ADConvPoint as Net

        model = Net(self.input_channels, self.output_channels, config).float()
        return model


class Trainer():

    def __init__(self, dataContainer, net, config, folderName=None):

        self.N_LABELS = len(dataContainer.getLabels())
        self.config = config
        # define the save directory
        time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.save_dir = os.path.join(
            folderName, 
            "Experiment_{}".format(time_string))
        # setting stuff for trainer
        if config["cuda"]:
            net.cuda()
        print("Number of parameters", self.count_parameters(net))
        config["parameters_count"] = self.count_parameters(net)
        self.train_loader, self.test_loader = dataContainer.getDataLoader(
            numPts=config['npoints'],
            threads=0,
            iterPerShape=1,
            batchSize=config['batchsize']
        )
        self.test_data = dataContainer.getTestData()
        self.test_labels = dataContainer.getTestLabels()
        self.train_data = dataContainer.getTrainData()
        self.train_labels = dataContainer.getTrainLabels()
        self.optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-6)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [7,15, 20])
        self.net = net
        # ad parameters
        self.R = torch.tensor(self.config['R'])   # radius R initialized with 0 by default.
        self.c = torch.tensor(self.config['c']) if self.config['c'] is not None else None
        self.reg_weight = 1
        self.nu = self.config['nu']
        self.soft_bound = config['soft_bound']
        self.warm_up_n_epochs = config['warm_up_n_epochs']
        self.noise_reg = config['noise_reg']
        self.config = config

        self.rocs = []
        self.aucs = []

    def ruffLoss(self, outputs, reg_outputs):

        dist = torch.sum((outputs - self.c) ** 2, dim=1)
        #  self vs hard svdd loss
        if self.soft_bound:
            scores = dist - self.R ** 2
            loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
        else:
            loss = torch.mean(dist)
        ## --------------------------------------------------------
        ## Regularization to prevent mode collapse
        if self.noise_reg:
            alpha = 0.8
            beta = 0.5
            reg_loss = nn.BCEWithLogitsLoss(reduction='sum')
            random_target = torch.empty(reg_outputs.shape, requires_grad=False).random_(0,2)
            if self.config["cuda"]:
                random_target = random_target.cuda()
            self.reg_weight = alpha*self.reg_weight + beta*(1-alpha)*(loss/reg_loss(reg_outputs,random_target)).item()
            loss += self.reg_weight*reg_loss(reg_outputs,random_target)
        ## --------------------------------------------------------
        return loss

    def get_radius(self, dist: torch.Tensor, nu: float):
        """Optimally solve for radius R via the (1-nu)-quantile of distances."""
        return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)

    def init_logs(self):

        print('\nCreating log directory...\n')
        print(self.save_dir + "\n")
        os.makedirs(self.save_dir, exist_ok=True)
        logFile = open(os.path.join(self.save_dir, "logs.txt"), "w")
        configFile = open(os.path.join(self.save_dir, "config.txt"), "w")
        configFile.write(str(self.config))
        return logFile

    def init_center_c(self, train_loader, net, eps=0.1):

        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim)
        if self.config["cuda"]:
            c = c.cuda()
        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                pts, features, _2, _3  = data
                if self.config["cuda"]:
                    features, pts = features.cuda(), pts.cuda()
                outputs, _ = net(features, pts)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)
        c /= n_samples
        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c


    def count_parameters(self, model):
        parameters = model.parameters()
        return sum(p.numel() for p in parameters if p.requires_grad)


    def train(self, epoch_nbr=100):

        f = self.init_logs()
        self.c = self.init_center_c(self.train_loader, self.net)
        print("\n center is: \n")
        print(self.c)
        for epoch in range(epoch_nbr):

            self.net.train()
            train_aloss = self.apply(epoch, training=True)
            # No evaluation, no radius adjustment if soft boundary loss is used
            if (epoch >= self.warm_up_n_epochs):
                self.net.eval()
                with torch.no_grad():
                    test_auc, test_aa, roc, self.outputs = self.apply(epoch, training=False)
                    self.rocs.append(roc)
                    self.aucs.append(float(test_auc))
            # save network
                torch.save(self.net.state_dict(), os.path.join(self.save_dir, "state_dict.pth"))
            # write the logs
                f.write(str(epoch)+",")
                f.write(train_aloss+",")
                f.write(test_auc+"\n")
                f.flush()
            else:
                f.write(str(epoch)+",")
                f.write(train_aloss + ",")
                f.write("NaN"+"\n")
                f.flush()
            self.scheduler.step()
        f.close()

    def apply(self, epoch, training=False):
        error = 0
        cm = np.zeros((2, 2))
        n_samples = 0
        if training:
            predictions = np.zeros((self.train_data.shape[0]), dtype=int)
            t = tqdm(self.train_loader, desc="Epoch " + str(epoch), ncols=130)
            for pts, features, targets, indices in t:
                # self.train_labels
                if self.config['cuda']:
                    features = features.cuda()
                    pts = pts.cuda()

                self.optimizer.zero_grad()
                outputs, reg_outputs = self.net(features, pts)
                loss = self.ruffLoss(outputs, reg_outputs)
                self.last_loss = loss
                torch.autograd.set_detect_anomaly(True)
                loss.backward(retain_graph=False)
                self.optimizer.step()

                # update the radius after warm up epochs
                if (epoch >= self.warm_up_n_epochs):
                    dist = torch.sum((outputs - self.c) ** 2, dim=1)
                    self.R.data = torch.tensor(self.get_radius(dist, self.nu))

                error += loss.item()
                n_samples += 1
                # scores
                aloss = "{:.3e}".format(error / n_samples)
                t.set_postfix(ALoss=aloss)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                scores = dist - self.R
                scores_np = scores.cpu().detach().numpy()
                scores_np = (scores_np >= self.R.numpy()).astype(int)
                for i in range(indices.size(0)):
                    predictions[indices[i]] = scores_np[i]

            return aloss

        else:
            predictions = np.zeros((self.test_data.shape[0]), dtype=int)
            distances = np.zeros((self.test_data.shape[0]), dtype=float)
            t = tqdm(self.test_loader, desc="  Test " + str(epoch), ncols=100)

            for pts, features, targets, indices in t:
                if self.config['cuda']:
                    features = features.cuda()
                    pts = pts.cuda()
                    targets = targets.cuda()
                outputs, reg_outputs = self.net(features, pts)
                targets = targets.view(-1)
                loss = self.ruffLoss(outputs, reg_outputs=reg_outputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                scores = dist - self.R
                scores_np = scores.cpu().detach().numpy()
                pred = (scores_np >= self.R.numpy()).astype(int)
                for i in range(indices.size(0)):
                    predictions[indices[i]] = pred[i]
                    distances[indices[i]] = scores_np[i]
                error += loss.item()
                if self.config['ntree'] == 1:
                    pred_labels = pred
                    cm_ = confusion_matrix(targets.cpu().numpy(), pred_labels, labels=list(range(2)))
                    cm += cm_
                    # scores
                    aa = "{:.4f}".format(metrics.stats_accuracy_per_class(cm)[0])
                    aiou = "{:.4f}".format(metrics.stats_iou_per_class(cm)[0])
                    aloss = "{:.4e}".format(error / cm.sum())
                    t.set_postfix( AA=aa, AIOU=aiou, ALoss=aloss)

            cm = confusion_matrix(self.test_labels, predictions, labels=list(range(2)))
            normAcc = "{:.5f}".format(cm[0,0] / (-(self.test_labels -1)).sum())
            anomAcc = "{:.5f}".format(cm[1,1] / self.test_labels.sum())
            aa = "{:.5f}".format(metrics.stats_accuracy_per_class(cm)[0])
            aiou = "{:.5f}".format(metrics.stats_iou_per_class(cm)[0])
            auc = "{:.4f}".format(metrics.roc_auc_score(self.test_labels, distances))
            print("Predictions",  "AUC", auc)
                                  #"AA", aa,
                                  #"IOU", aiou,
                                  #"normAcc", normAcc,
                                  #"anomAcc", anomAcc)
            roc_fpr, roc_tpr, roc_thr = metrics.roc_curve(self.test_labels, distances)
            return auc, aa, [roc_fpr, roc_tpr, roc_thr], distances
