from tqdm import tqdm

class Trainer():
    
    def __init__(self):
        pass
    
    def loss(self, output):
        pass
    
    def compute_train_metrics(self, output_vector, targets):
        pass
    
    def compute_test_metrics(self, output_vector, targets):
        pass
    
    def train(self, epoch_nbr):
        pass

    def apply(self, epoch, training=False):
        if training:

            t = tqdm(self.train_loader, desc="Epoch " + str(epoch), ncols=130)
            for pts, features, targets, indices in t:
                if self.config['cuda']:
                    features = features.cuda()
                    pts = pts.cuda()
                    self.optimizer.zero_grad()
                    output_vector = self.net(features, pts)
                    loss = self.loss(output_vector)
                    self.last_loss = loss
                    loss.backward()
                    self.optimizer.step()
                    self.compute_train_metrics()
        else:


