import torch
import torch.nn as nn
from typing import Dict

class SiameseNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # input of size 105*105
            nn.Conv2d(1, 64, kernel_size=10),  # 64@96*96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64@48*48
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),  # 128@42*42
            nn.MaxPool2d(2),  # 128@21*21
            nn.Conv2d(128, 128, 4),
            nn.ReLU(),  # 128@18*18
            nn.MaxPool2d(2),  # 128@9*9
            nn.Conv2d(128, 256, 4),
            nn.ReLU(),  # 256@6*6
        )
        self.liner = nn.Sequential(nn.Linear(9216, 1024))
        self.siam_out = nn.Sequential(nn.Linear(1024, 1), nn.Sigmoid())
        self.sim_fc = nn.Sequential(nn.Linear(6, 4), nn.ReLU(),
                                    nn.Linear(4, 3), nn.ReLU(),
                                    nn.Linear(3, 1), nn.Sigmoid())

        self._step = ""

    def set_step_pretrain_siam(self):
        self._step = "pretrain_siamese"

    def set_step_pretrain_fc(self):
        self._step = "pretrain_fc"

    def set_step_train_fine_tune(self):
        self._step = "train_full"

    def _conv_branch(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.liner(x)
        return x

    def _pretrain_siamese(self, data):
        x0 = self._conv_branch(data['x0'])
        x1 = self._conv_branch(data['x1'])
        d0_1 = self.siam_out(torch.abs(x0 - x1))
        return d0_1

    def _pretrain_fully_connected_part(self, data):
        with torch.no_grad():
            d0_1, d0_1a0, x0, x1, x1_a0 = self._image_distances(data)
        out = self._fc(data, d0_1, d0_1a0)
        return out

    def _image_distances(self, data: Dict):
        try:
            x0 = data['c0']
        except KeyError:
            x0 = self._conv_branch(data['x0'])
        try:
            x1 = data['c1']
        except KeyError:
            x1 = self._conv_branch(data['x1'])

        try:
            x1_a0 = data['c1_0']
        except KeyError:
            x1_a0 = self._conv_branch(data['x1_a0'])


        d0_1a0 = self.siam_out(torch.abs(x0 - x1_a0))
        d0_1 = self.siam_out(torch.abs(x0 - x1))

        return d0_1, d0_1a0, x0, x1, x1_a0

    def _fc(self, data, d0_1, d0_1a0):
        ar = data['ar'].to(torch.float32).flatten().unsqueeze(1)
        d = data['log_d'].to(torch.float32).flatten().unsqueeze(1)
        area_0 = data['area_0'].to(torch.float32).flatten().unsqueeze(1)
        t = data['t'].to(torch.float32).flatten().unsqueeze(1)
        dissim_data = torch.cat((d0_1, d0_1a0, ar, d, area_0, t), 1)
        out = self.sim_fc(dissim_data)
        return out

    def _train_full(self, data):
        d0_1, d0_1a0, x0, x1, x1_a0 = self._image_distances(data)
        out = self._fc(data, d0_1, d0_1a0)
        return out

    def _fast_inference(self, data):
        d0_1, d0_1a0, x0, x1, x1_a0 = self._image_distances(data)
        out = self._fc(data, d0_1, d0_1a0)
        return out, x0, x1, x1_a0

    def forward(self, data):
        if self._step == 'pretrain_siamese':
            return self._pretrain_siamese(data)
        elif self._step == 'pretrain_fc':
            return self._pretrain_fully_connected_part(data)
        elif self._step == 'train_full':
            return self._train_full(data)
        else:
            return self._fast_inference(data)
