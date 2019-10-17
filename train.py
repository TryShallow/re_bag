import os
import time
import torch
import model
import model_glove_elmo
import torch.nn as nn

from torch import optim
from dataset_glove_elmo import MyDataset
from torch.utils.data import DataLoader


class Trainer(object):
    def __init__(self, train_data, dev_data, logger):
        self.logger = logger

        self.batch_size = 20
        self.num_epochs = 5000
        self.gpus = ['cuda:0']
        self.model_path = 'saved_models'
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.train_data_loader = DataLoader(train_data, self.batch_size)
        self.dev_data_loader = DataLoader(dev_data, self.batch_size)
        self.model = model_glove_elmo.Model(self.gpus[0])
        self.load_parameters()
        self.model.to(self.gpus[0])

        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def load_parameters(self, model_file=""):
        if model_file != "":
            self.model.load_state_dict(torch.load(self.model_path + '/' + model_file))
            self.logger.info('Load parameters file ' + model_file)
        else:
            bc = 0.0
            for file in os.listdir(self.model_path):
                if file.startswith('wiki_ge_'):
                    ac = float(file.split('_')[-1])
                    if bc < ac:
                        bc = ac
            if bc > 0.0:
                self.load_parameters('wiki_ge_' + str(bc))

    def train(self):
        self.model.train()
        best_acc = 0.0
        save_model_prefix = os.path.join(self.model_path, "wiki_ge_")
        for epoch in range(self.num_epochs):
            self.logger.info("Epoch %d/%d" % (epoch + 1, self.num_epochs))
            start_time = time.time()
            for batch in self.train_data_loader:
                output = self.model(MyDataset.to(batch, self.gpus[0]))
                self.model.zero_grad()
                loss = self._calc_loss(output, batch)
                loss.backward()
                self.optimizer.step()

            time_diff = time.time() - start_time
            self.logger.info("epoch %d time consumed: %dm%ds." % (epoch + 1, time_diff // 60, time_diff % 60))

            # evaluate model
            cur_acc = self.eval_dev(self.dev_data_loader)
            self.model.train()
            self.logger.info("Current accuracy: %.3f" % cur_acc)
            if cur_acc > best_acc:  # and epoch > 10:
                save_filename = save_model_prefix + str(cur_acc)
                torch.save(self.model.state_dict(), save_filename)
                best_acc = cur_acc

    def eval(self):
        self.model.eval()

    def eval_dev(self, dev_data_loader):
        self.model.eval()
        correct_count = 0
        total_count = 0
        for batch in dev_data_loader:
            output = self.model(MyDataset.to(batch, self.gpus[0]))
            pred = torch.argmax(output, 1)
            correct_count += (pred.cpu().detach().numpy() == batch['answer_candidates_id_mb'].numpy()).sum()
            total_count += len(batch['query_length_mb'])
        return float(correct_count) / total_count


    def _calc_loss(self, output, batch):
        if len(self.gpus) == 1:
            cross_entropy = nn.CrossEntropyLoss(reduction='mean')
            return cross_entropy(output, batch['answer_candidates_id_mb'].to(self.gpus[0]))
