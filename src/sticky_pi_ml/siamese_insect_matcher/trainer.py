import json
import os
import torch
import torch.nn as nn
import logging
import numpy as np

from sticky_pi_ml.trainer import BaseTrainer
from sticky_pi_ml.siamese_insect_matcher.ml_bundle import MLBundle
from sticky_pi_ml.siamese_insect_matcher.model import SiameseNet
from sticky_pi_ml.siamese_insect_matcher.predictor import Predictor


class Trainer(BaseTrainer):
    def __init__(self, ml_bundle: MLBundle):
        super().__init__(ml_bundle)
        self._save_every = ml_bundle.config['CHECKPOINT_PERIOD']
        self._net = SiameseNet()
        self._config = self._ml_bundle.config

    def resume_or_load(self, resume=True):
        weights = self._ml_bundle.weight_file
        if resume:
            if not os.path.exists(weights):
                logging.warning("%s not found. Training from start" % weights)
                self.resume_or_load(False)
            else:
                logging.info("Starting from checkpoint")
                self._net.load_state_dict(torch.load(weights))
        else:
            logging.info("Training from start")
            if not os.path.isdir(os.path.dirname(weights)):
                os.mkdir(os.path.dirname(weights))

    def train_step(self, train_loader,  base_lr, n_rounds, step_name):
        lr_decay = self._config['GAMMA']
        # lr_momentum = self._config['LR_MOMENTUM']

        optimizer = torch.optim.Adam(self._net.parameters(), 1)
        loss_func = nn.BCELoss()

        running_loss = None
        round = 0
        while True:
            for i, (data, labels) in enumerate(train_loader, 0):

                if round >= n_rounds:
                    return

                lr = base_lr * lr_decay ** float(round)
                logging.info('Step: %s; Round: %i / %i; LR: %f' % (step_name, round + 1, n_rounds, lr))

                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                # add singleton dimension after. Now, we have a batch dimension
                labels = labels.unsqueeze(1).float()
                optimizer.zero_grad()
                f = self._net(data)
                loss = loss_func(f, labels)

                y = f.detach().numpy().flatten()
                x = labels.detach().numpy().flatten()
                y = np.round(y)

                loss.backward()
                optimizer.step()
                if running_loss is None:
                    running_loss = loss.item()
                running_loss = running_loss * 0.95 + loss.item() * 0.05

                print('% right', np.mean(x == y), running_loss, loss.item())
                round += 1
                if round % self._save_every == self._save_every - 1:
                    torch.save(self._net.state_dict(), self._ml_bundle.weight_file)


    def train(self):
        self._net.train()
        if self._config['DEVICE'] == 'cuda':
            self._net = self._net.cuda()

        train_loader = self._ml_bundle.dataset.get_torch_data_loader('train')
        logging.info('N pairs: %i' % len(train_loader.dataset))

        self._net.set_step_pretrain_siam()
        self.train_step(train_loader, self._config['SIAM_BASE_LR'], self._config['SIAM_ROUNDS'], 'Siamese')
        self._net.set_step_pretrain_fc()
        self.train_step(train_loader,  self._config['DIST_AR_BASE_LR'], self._config['DIST_AR_ROUNDS'], 'Dist AR')
        self._net.set_step_train_fine_tune()
        self.train_step(train_loader,  self._config['FINAL_BASE_LR'],  self._config['FINAL_ROUNDS'], 'FULL')

    def validate(self, predictor: Predictor, out_dir: str = None):

        dl = self._ml_bundle.dataset.get_torch_data_loader('val', shuffle=False)
        out = []

        for i, (data, label) in enumerate(dl, 0):

            f = predictor.match_torch_batch(data)
            # label = label.float().squeeze_()
            for pred, gt in zip(f.detach().numpy().flatten(), label.detach().numpy().flatten()):
                print(round(pred, 2), gt)
                out.append({'pred':float(pred), 'gt':float(gt)})
        with open(os.path.join(out_dir, 'result.json'), 'w') as file:
            file.write(json.dumps(out))
