import glob
from math import log10
import os
import torch
import cv2
import numpy as np
from sticky_pi_ml.dataset import BaseDataset
import logging
from torch.utils.data import Dataset as TorchDataset
# DataLoader
from torchvision.transforms import ToTensor, Compose
from detectron2.data import transforms
from sticky_pi_ml.annotations import Annotation
from sticky_pi_ml.siamese_insect_matcher.siam_svg import SiamSVG
from sticky_pi_ml.siamese_insect_matcher.model import SiameseNet
from sticky_pi_ml.utils import pad_to_square, detectron_to_pytorch_transform, iou, md5
import random

from typing import List

to_tensor_tr = ToTensor()


class DataEntry(object):
    _im_dim = 105  # fixme. could be inferred from conf / net?
    _default_transform = [to_tensor_tr] * 2

    def __init__(self, a0: Annotation,
                 a1: Annotation,
                 im1: np.ndarray, n_pairs:int,
                 data_transforms=None, dist_transform=None,
                 # a hash of the net used and to cache the result of convolution,
                 # when mathching all against all, we really don';t need to comput the full convolution for each layer!
                 net_for_cache: SiameseNet = None):

        if data_transforms is None:
            data_transforms = self._default_transform

        try:
            self._c0 = a0.cached_conv[net_for_cache]
            self._x0 = None
        except KeyError:
            self._x0 = DataEntry.make_array_for_annot(a0)
            self._x0 = data_transforms[0](self._x0)
            self._c0 = None

        try:
            self._c1 = a1.cached_conv[net_for_cache]
            self._x1 = None
        except KeyError:
            self._x1 = DataEntry.make_array_for_annot(a1)
            self._x1 = data_transforms[1](self._x1)
            self._c1 = None

        try:
            self._c1_0 = a0.cached_conv[(net_for_cache, a1.datetime)]
            self._x1_a0 = None
        except KeyError:
            # view of a0 in im1 => info about whether insect has moved! (if so, no insect in im0 * a1)
            self._x1_a0 = DataEntry.make_array_for_annot(a0, source_array=im1)
            self._x1_a0 = data_transforms[0](self._x1_a0)
            self._c1_0 = None

        dist = abs(a0.center - a1.center)
        if dist_transform is not None:
            dist = dist_transform(dist)

        dist = log10(dist + 1)
        self._log_dist = torch.Tensor([dist])
        self._n_pairs = None

        if n_pairs is not None:
            self._n_pairs = torch.Tensor([n_pairs])

        self._ar = abs(log10(a1.area) - log10(a0.area))
        self._ar = torch.Tensor([self._ar])
        self._area_0 = torch.Tensor([log10(a0.area)])


    def as_dict(self, add_dim=False):
        out = {'x0': self._x0,
               'x1': self._x1,
               'x1_a0': self._x1_a0,
               'c0': self._c0,
               'c1': self._c1,
               'c1_0': self._c1_0,
               'log_d': self._log_dist,
               'ar': self._ar,
               'area_0': self._area_0,
               }

        if add_dim:
            with torch.no_grad():
                for k in out.keys():
                    if out[k] is not None and k not in {'c0', 'c1', 'c1_0'}:
                        out[k].unsqueeze_(0)
        todel = [k for k in out.keys() if out[k] is None]
        for t in todel:
            del out[t]

        return out

    @classmethod
    def make_array_for_annot(cls, a, source_array: np.array = None, to_tensor=False):
        arr = a.subimage(masked=True, source_array=source_array)
        arr = pad_to_square(arr, cls._im_dim)
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        if to_tensor:
            arr = to_tensor_tr(arr)
            arr.unsqueeze_(0)
        return arr


class OurTorchDataset(TorchDataset):
    def __init__(self, data_dicts, augment=True):

        self._augment = augment
        if self._augment:
            self._transforms = [Compose([
                detectron_to_pytorch_transform(transforms.RandomBrightness)(0.75, 1.25),
                detectron_to_pytorch_transform(transforms.RandomContrast)(0.75, 1.25),
                detectron_to_pytorch_transform(transforms.RandomFlip)(horizontal=True, vertical=False),
                detectron_to_pytorch_transform(transforms.RandomFlip)(horizontal=False, vertical=True),
                detectron_to_pytorch_transform(transforms.RandomRotation)(angle=[0, 90, 180, 270],
                                                                          sample_style='choice'),
                to_tensor_tr,
            ])] * 2
            self._dist_transform = np.random.exponential
        else:
            self._transforms = [Compose([to_tensor_tr])] * 2
            self._dist_transform = None

        self._negative_data_pairs, self._positive_data_pairs = [], []
        for d in data_dicts:
            if d['label'] == 1:
                self._positive_data_pairs.append(d)
            else:
                self._negative_data_pairs.append(d)

    def _get_one(self, item: int):
        if item >= len(self._positive_data_pairs):
            entry = self._negative_data_pairs[item % len(self._positive_data_pairs)]
        else:
            entry = self._positive_data_pairs[item]

        out = DataEntry(**entry['data'], data_transforms=self._transforms, dist_transform=self._dist_transform)
        return out.as_dict(), entry['label']

    def __iter__(self):
        for i in range(self.__len__()):
            yield self._get_one(i)

    def __getitem__(self, item, prob_neg=0.50):
        if random.random() < prob_neg:
            return self._get_one(random.randint(0, len(self._negative_data_pairs)))
        else:
            return self._get_one(random.randint(len(self._positive_data_pairs), self.__len__()))



    def __len__(self):
        return len(self._negative_data_pairs) + len(self._positive_data_pairs)


class Dataset(BaseDataset):
    # fixme this can be a config var
    _max_iou = 0.9  # we don't use for training any pair that has iou > max_iou

    def __init__(self, data_dir, config, cache_dir):
        super().__init__(data_dir, config, cache_dir)

    def _prepare(self):
        input_img_list = sorted(glob.glob(os.path.join(self._data_dir, '*.svg')))
        data = self._serialise_imgs_to_dicts(input_img_list)
        while len(data) > 0:
            entry = data.pop()
            if entry['md5'] > self._md5_max_training:
                self._validation_data.append(entry)
            else:
                self._training_data.append(entry)
        # print(len(self._training_data))
        # print(len(self._val_data))

    def _serialise_imgs_to_dicts(self, input_img_list: List[str]):
        pos_pairs = []
        neg_pairs = []
        iou_max_n_discarded = 0

        for s in sorted(input_img_list):
            logging.info('Serializing: %s' % os.path.basename(s))
            ssvg = SiamSVG(s)
            md5_sum = md5(s)
            a0_annots = []
            a1_annots = []
            for a0, a1 in ssvg.annotation_pairs:
                a0_annots.append(a0)
                a1_annots.append(a1)

            for i, a0 in enumerate(a0_annots):
                for j, a1 in enumerate(a1_annots):
                    im1 = a1.parent_image.read()

                    if i < j:
                        continue

                    iou_val = iou(a0.polygon, a1.polygon)
                    data = {'a0': a0, 'a1': a1, 'im1': im1, 'n_pairs': len(ssvg.annotation_pairs)}
                    if iou_val > self._max_iou:
                        iou_max_n_discarded += 1
                        continue
                    elif i == j:
                        pos_pairs.append({'data': data, 'label': 1, 'md5': md5_sum})
                    else:
                        neg_pairs.append({'data': data, 'label': 0, 'md5': md5_sum})

        logging.info('Serialized: %i positive and %i negative. Discarded %i matches with iou>max_iou' %
                     (len(pos_pairs), len(neg_pairs), iou_max_n_discarded))
        return pos_pairs + neg_pairs

        # n_neg = int(prob_neg * len(pos_pairs) / (1-prob_neg))
        # n_neg = min(len(neg_pairs), n_neg)
        # neg_pairs = random.sample(neg_pairs, n_neg)
        # selected_pairs = pos_pairs + neg_pairs
        # random.shuffle(selected_pairs)
        # out = []
        # for a0, a1, im1, score, lp in selected_pairs:
        #     out.append(DataEntry(a0, a1, im1, score, lp, self._transforms, self._dist_transform))
        # DataEntry(a0, a1, im1, score, lp, self._transforms, self._dist_transform)

    def get_torch_data_loader(self, subset='train', shuffle=True):
        assert subset in {'train', 'val'}, 'subset should be either "train" or "val"'
        augment = subset == 'train'
        to_load = self.get_torch_dataset(subset, augment=augment)

        out = torch.utils.data.DataLoader(to_load,
                                          batch_size=self._config['IMS_PER_BATCH'],
                                          shuffle=shuffle,
                                          num_workers=self._config['N_WORKERS'])
        return out

    def get_torch_dataset(self, subset='train', augment=False):
        assert subset in {'train', 'val'}, 'subset should be either "train" or "val"'
        data = self._training_data if subset == 'train' else self._validation_data
        return OurTorchDataset(data, augment=augment)

    def visualise(self, subset='train', augment=False, interactive=True):
        import cv2
        buff = None
        for dt in self.get_torch_dataset(subset, augment=augment):
            d, label = dt

            im0 = d['x0']
            im1 = d['x1']
            im1_a0 = d['x1_a0']
            if buff is None:
                w = im0.shape[2] * 3
                h = im0.shape[1]
                buff = np.zeros((h, w, im0.shape[0]), im0.numpy().dtype)

            buff[:, 0:im0.shape[2], :] = np.moveaxis(im0.numpy(), 0, -1)
            buff[:, im0.shape[2]:im0.shape[2] * 2, :] = np.moveaxis(im1.numpy(), 0, -1)
            buff[:, im0.shape[2] * 2:, :] = np.moveaxis(im1_a0.numpy(), 0, -1)

            if interactive:
                cv2.imshow('s0', buff)
                cv2.waitKey(-1)
            else:
                assert isinstance(buff, np.ndarray)

