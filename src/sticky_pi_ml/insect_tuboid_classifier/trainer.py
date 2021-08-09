import datetime
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import numpy as np
import os
import logging

from sticky_pi_ml.trainer import BaseTrainer
from sticky_pi_ml.insect_tuboid_classifier.ml_bundle import MLBundle
from sticky_pi_ml.insect_tuboid_classifier.model import make_resnet


class Trainer(BaseTrainer):
    def __init__(self, ml_bundle: MLBundle):
        """
        See https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
        :param ml_bundle:
        """
        super().__init__(ml_bundle)
        self._save_every = ml_bundle.config['CHECKPOINT_PERIOD']
        self._net = None
        self._config = self._ml_bundle.config

    def resume_or_load(self, resume=True):
        weights = self._config['WEIGHTS']
        if resume:
            if not os.path.exists(weights):
                logging.warning("%s not found. Training from start" % weights)
                self.resume_or_load(False)
            else:
                logging.info("Starting from checkpoint")
                self._net = make_resnet(pretrained=False, n_classes=self._ml_bundle.dataset.n_classes)
                map = None if torch.cuda.is_available() else 'cpu'
                self._net.load_state_dict(torch.load(weights, map_location=map))
        else:
            logging.info("Training from start")
            self._net = make_resnet(pretrained=True, n_classes=self._ml_bundle.dataset.n_classes)

    def train(self):
        base_lr = self._config['BASE_LR']
        lr_momentum = self._config['LR_MOMENTUM']
        n_rounds = self._config['N_ROUNDS']
        gamma = self._config['GAMMA']
        n_classes = self._ml_bundle.dataset.n_classes

        dataloaders_dict = {x: self._ml_bundle.dataset.get_torch_data_loader(subset=x, shuffle=s)
                            for x, s in zip(['train', 'val'], [True, False])}

        # Detect if we have a GPU available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Send the model to GPU, if available
        model = self._net.to(device)
        params_to_update = model.parameters()
        optimizer = optim.SGD(params_to_update, lr=base_lr, momentum=lr_momentum)
        criterion = nn.CrossEntropyLoss()

        running_loss = None
        training_round = 0
        to_validate = False

        self._validate(model, dataloaders_dict['val'], criterion, device, n_classes, 0, base_lr)

        while True:
            all_losses = []
            all_accu = []
            model.train()

            for i, (inputs, labels) in enumerate(dataloaders_dict['train'], 0):
                if training_round >= n_rounds:
                    return
                optimizer.zero_grad()
                lr = base_lr * gamma ** float(training_round)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                for k, v in inputs.items():
                    if torch.is_tensor(v):
                        inputs[k] = inputs[k].to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                loss.backward()
                optimizer.step()

                # statistics
                labels_d = labels.detach()
                preds_d = preds.detach()

                if running_loss is None:
                    running_loss = loss.item()

                running_loss = running_loss * 0.95 + loss.item() * 0.05

                all_losses.append(loss.item())
                all_accu.append(torch.mean((labels_d == preds_d).float()).item())

                # print('TRAINING: Round %i; Accuracy %f;Running loss %f; Loss %f, LR %f' % (
                #     training_round, torch.mean((labels_d == preds_d).float()).item(), running_loss, loss.item(), lr))


                training_round += 1
                if training_round % self._save_every == self._save_every - 1:
                    to_validate = True
                    break

            if to_validate:
                print('TRAINING:',
                      training_round,
                      lr,
                      np.mean(all_accu),
                      np.mean(all_losses),
                      str(datetime.datetime.now()))
                self._validate(model, dataloaders_dict['val'], criterion, device, n_classes, training_round, lr)
                print('SNAPSHOOTING', str(datetime.datetime.now()))
                torch.save(self._net.state_dict(), self._config['WEIGHTS'])
                to_validate = False

    def _validate(self, model, data_loader, criterion, device, n_classes,training_round, lr):
        model.eval()
        with torch.no_grad():
            epoch_labels = []
            epoch_preds = []
            all_losses = []

            for inputs, labels in data_loader:

                for k, v in inputs.items():
                    if torch.is_tensor(v):
                        inputs[k] = inputs[k].to(device)

                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                # # statistics
                labels_d = labels.detach()
                preds_d = preds.detach()

                all_losses += [loss.item()] * len(labels)
                epoch_labels.extend(labels_d)
                epoch_preds.extend(preds_d)

            print('VALIDATION:',
                  training_round,
                  lr,
                  np.mean(np.array(epoch_labels) == np.array(epoch_preds)),
                  np.mean(all_losses),
                  str(datetime.datetime.now()))
            try:
                from sklearn.metrics import confusion_matrix, classification_report
                print(confusion_matrix(epoch_labels, epoch_preds, labels=np.arange(0, n_classes)))
                print(classification_report(epoch_labels, epoch_preds, labels=np.arange(0, n_classes)))
            except ImportError as e:
                pass




