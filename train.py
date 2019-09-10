import time

from model import Model
from torch.utils.data import DataLoader


class Trainer(object):
    def __init__(self, train_data, dev_data):
        self.batch_size = 2
        self.num_epochs = 50

        self.train_data_loader = DataLoader(train_data, self.batch_size)
        self.dev_data_loader = DataLoader(dev_data, self.batch_size)
        self.model = Model()

    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            print("Epoch %d/%d" % (epoch + 1, self.num_epochs))
            start_time = time.time()
            for batch in self.train_data_loader:
                outputs = self.model(batch)
                self.model.zero_grad()
                break

            time_diff = time.time() - start_time
            print("epoch %d time consumed: %dm%ds." % (epoch + 1, time_diff // 60, time_diff % 60))

    def eval(self):
        self.model.eval()