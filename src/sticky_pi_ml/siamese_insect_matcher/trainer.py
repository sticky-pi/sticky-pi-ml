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


    def _validation_loss(self, val_loader):
        loss_func = nn.BCELoss()
        loss_sum = 0.0
        accuracy_sum = 0.0
        n_elements = 0
        for i, (data, labels) in enumerate(val_loader, 0):
            with torch.no_grad():
                labels = labels.unsqueeze(1).float()
                f = self._net(data)
                loss = loss_func(f, labels)

                y = f.detach().numpy().flatten()
                x = labels.detach().numpy().flatten()
                y = np.round(y)
                accuracy_sum += np.mean(x == y) * len(labels)
                loss_sum += loss.item() * len(labels)
                n_elements += len(labels)

        return accuracy_sum/n_elements, loss_sum/n_elements


    def train_step(self, train_loader, val_loader,  base_lr, n_rounds, step_name):
        lr_decay = self._config['GAMMA']
        # lr_momentum = self._config['LR_MOMENTUM']

        optimizer = torch.optim.Adam(self._net.parameters(), 1)
        loss_func = nn.BCELoss()

        running_loss = None
        round = 0
        logging.info('Starting initial validation')
        val_accuracy, val_loss = self._validation_loss(val_loader)
        logging.info('Starting training')

        while True:
            for i, (data, labels) in enumerate(train_loader, 0):

                if round >= n_rounds:
                    return

                lr = base_lr * lr_decay ** float(round)
                # logging.info('Step: %s; Round: %i / %i; LR: %f' % (step_name, round + 1, n_rounds, lr))

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
                logging.info(f'Step: {step_name}; Round:  {round + 1} / {n_rounds}; LR: {lr:e}; accuracy: {np.mean(x == y):.5f}; '
                             f' running_loss: {running_loss:.5f}, loss: {loss.item():.5f}, val_accuracy: {val_accuracy:.5f}, val_loss: {val_loss:.5f}')

                round += 1
                if round % self._save_every == self._save_every - 1:
                    torch.save(self._net.state_dict(), self._ml_bundle.weight_file)
                    val_accuracy, val_loss = self._validation_loss(val_loader)

    def train(self):
        self._net.train()
        if self._config['DEVICE'] == 'cuda':
            self._net = self._net.cuda()

        train_loader = self._ml_bundle.dataset.get_torch_data_loader('train')
        val_loader = self._ml_bundle.dataset.get_torch_data_loader('val', shuffle=False)
        logging.info('N pairs: %i' % len(train_loader.dataset))

        self._net.set_step_pretrain_siam()
        self.train_step(train_loader, val_loader,  self._config['SIAM_BASE_LR'], self._config['SIAM_ROUNDS'], 'Siamese')
        self._net.set_step_pretrain_fc()
        self.train_step(train_loader, val_loader,  self._config['DIST_AR_BASE_LR'], self._config['DIST_AR_ROUNDS'], 'Dist AR')
        self._net.set_step_train_fine_tune()
        self.train_step(train_loader, val_loader,  self._config['FINAL_BASE_LR'],  self._config['FINAL_ROUNDS'], 'FULL')

    def validate(self, predictor: Predictor, out_dir: str = None):

        os.makedirs(out_dir, exist_ok=True)

        dl = self._ml_bundle.dataset.get_torch_data_loader('val', shuffle=False)
        out = []
        tp, tn, fp, fn = 0, 0, 0, 0

        for i, (data, label) in enumerate(dl, 0):

            f = predictor.match_torch_batch(data)
            # label = label.float().squeeze_()
            for pred, gt in zip(f.detach().numpy().flatten(), label.detach().numpy().flatten()):

                out.append({'pred': float(pred), 'gt':float(gt)})
                if int(gt) == 0:
                    if float(pred) < 0.5:
                        tn += 1
                    else:
                        fp += 1
                else: # gt > 1
                    if float(pred) < 0.5:
                        fn += 1
                    else:
                        tp += 1
            try:
                specificity = tn / (tn + fp)
            except ZeroDivisionError:
                specificity = "NaN"
            try:
                recall = tp / (tp + fn)
            except ZeroDivisionError:
                recall = "NaN"
            logging.info(f"Validation: {i+1} / {len(dl)}; specificity: {specificity}; recall: {recall}; N_total: {tn + fp + tp + fn}")

        with open(os.path.join(out_dir, 'results.json'), 'w') as file:
            file.write(json.dumps(out))
