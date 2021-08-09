import sqlite3
import os
import torch
from typing import List, Dict, Union
import logging
import numpy as np
import pandas as pd
from torch.utils.data import Dataset as TorchDataset
from torch import Tensor
from torchvision.transforms import ToTensor, Compose, Normalize
from detectron2.data import transforms

from sticky_pi_ml.dataset import BaseDataset
from sticky_pi_ml.utils import detectron_to_pytorch_transform
from sticky_pi_ml.insect_tuboid_classifier.taxonomy import TaxonomyMapper
from sticky_pi_ml.tuboid import TiledTuboid

to_tensor_tr = ToTensor()
normalize_tr = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class OurTorchDataset(TorchDataset):
    _n_shots_drawn = 6
    default_transform = Compose([to_tensor_tr, normalize_tr])

    def __init__(self, tuboids: List[Dict[str, Union[TiledTuboid, str, int]]], augment=True):
        """
        TODO fixme
        tuboids = [(name, {images  :  [img_path_1, ..., img_path_n]),
                           metadata: {n_images: N, taxonomy:(type, order, family, genus, species, extra)}}]

        """
        self._tuboids = tuboids
        self._augment = augment
        if self._augment:
            self._transforms = Compose([
                detectron_to_pytorch_transform(transforms.RandomBrightness)(0.5, 1.5),
                detectron_to_pytorch_transform(transforms.RandomContrast)(0.5, 1.5),
                detectron_to_pytorch_transform(transforms.RandomSaturation)(0.5, 1.5),
                detectron_to_pytorch_transform(transforms.RandomFlip)(horizontal=True, vertical=False),
                detectron_to_pytorch_transform(transforms.RandomFlip)(horizontal=False, vertical=True),
                detectron_to_pytorch_transform(transforms.RandomRotation)(angle=[0, 360], sample_style='range'),
                detectron_to_pytorch_transform(transforms.Resize)((224, 224)),
                to_tensor_tr,
                normalize_tr

            ])
        else:
            self._transforms = self.default_transform

    def __getitem__(self, item: int):
        tub = self._tuboids[item]['tuboid']
        # we pick the first shot of a tuboid and another n=self._n_shots -1
        out = OurTorchDataset.tiled_tuboid_to_dict(tub, self._transforms)
        return out, self._tuboids[item]['label']

    def __len__(self):
        return len(self._tuboids)

    @classmethod
    def tiled_tuboid_to_dict(cls, tuboid: TiledTuboid, im_transforms: Compose = None,
                             unsqueezed: bool = False) -> Dict[str, Tensor]:
        if im_transforms is None:
            im_transforms = cls.default_transform

        tile_ids_drawn = np.random.choice(tuboid.n_tiles - 1, cls._n_shots_drawn - 1) + 1
        tile_ids_drawn = [0] + tile_ids_drawn.tolist()

        scales = []
        arrays = []
        for t_id in tile_ids_drawn:
            tile_dict = tuboid.get_tile(t_id)
            scales.append(torch.Tensor([tile_dict['scale']]))
            array = im_transforms(tile_dict['array'])
            arrays.append(array)

        out = {'array': torch.stack(arrays),
               'scale': torch.stack(scales)}
        if unsqueezed:
            out = {k: v.unsqueeze(0) for k, v in out.items()}

        return out


class Dataset(BaseDataset):
    _annotations_filename = 'database.db'
    _annotations_table_name = 'ANNOTATIONS'

    def __init__(self, data_dir, config, cache_dir):
        """
        the annotation file is an sqlite file with a table named `ANNOTATIONS'.
        This table MUST have the fields:
        * ``tuboid_id``: a uid of a tuboid formatted like ``device.start.end.software_version.id`` (f8d61a5c.2020-07-08_22-00-00.2020-07-09_12-00-00.1607090908-d74d75f50086077dbab6b1dce8c02694.0003):
        * ``type``, ``order``, ``family`` ,``genus``,``species``. ``type`` is one of {``'Ambiguous'``,``'Background'``,``'Insecta'``}
        Individual tuboid data are in a directory named ``device.start.end.software_version/device.start.end.software_version.id``
        """
        super().__init__(data_dir, config, cache_dir)

        self._taxonomy_mapper = TaxonomyMapper(self._config['LABELS'])

    @property
    def n_classes(self):
        return self._taxonomy_mapper.n_classes

    @property
    def taxonomy_mapper(self):
        return self._taxonomy_mapper

    def _prepare(self):
        data = self._serialise_imgs_to_dicts()
        # we force each label to have a set proportion of the sample size for this label

        label_dict = {}

        while len(data) > 0:
            entry = data.pop()
            if not entry['label'] in label_dict.keys():
                label_dict[entry['label']] = []
            label_dict[entry['label']].append(entry)

        for label in label_dict.keys():
            label_dict[label].sort(key=lambda x: x['tuboid'].md5)

        prop_training = int(self._md5_max_training, 16) / int('ff', 16)

        for label in label_dict.keys():
            total_label = len(label_dict[label])
            for i, e in enumerate(label_dict[label]):
                if i < int(total_label * prop_training):
                    self._training_data.append(e)
                else:
                    self._validation_data.append(e)

        logging.info(f"ITC datasets")
        self._training_data.sort(key=lambda x: x['tuboid'].md5)
        logging.info(f"N_training : {len(self._training_data)}")
        self._validation_data.sort(key=lambda x: x['tuboid'].md5)
        logging.info(f"N_validation : {len(self._validation_data)}")

    def _serialise_imgs_to_dicts(self):
        sqlite_file = os.path.join(self._data_dir, self._annotations_filename)
        if not os.path.isfile(sqlite_file):
            raise FileNotFoundError(f'No database file found: {sqlite_file}')
        conn = sqlite3.connect(sqlite_file)
        try:
            annotations_df = pd.read_sql_query("select * from %s;" % self._annotations_table_name, conn)
        finally:
            conn.close()

        annotations_df.sort_values("datetime", inplace=True)

        # fixme should remove duplicates/ find consensus (e.g. the deepest)
        # for now, we drop duplicates, keeping the last annotation
        annotations_df.drop_duplicates(['tuboid_id'], keep='last', inplace=True)
        annotations_df.set_index('tuboid_id', inplace=True)
        l1 = len(annotations_df)
        # don't train with ambiguous data (e.g. stitching / segmentation errors)
        annotations_df = annotations_df[annotations_df.type != 'Ambiguous']
        l2 = len(annotations_df)
        logging.info(f"Dropping {l1 - l2} ambiguous rows")
        entries = []

        for t, r in annotations_df.iterrows():
            # t is the full tuboid id
            t = str(t)

            # full path to the parent directory of the tuboid (minus suffix)
            series_dir = os.path.join(self._data_dir, os.path.splitext(t)[0])

            if not os.path.isdir(series_dir):
                raise Exception("No series for tuboid %s. %s does not exist", t, series_dir)

            tuboid_dir = os.path.join(series_dir, t)
            if not os.path.isdir(tuboid_dir):
                raise Exception("No dir for tuboid %s %s. %s does not exist", t, tuboid_dir)

            tuboid = TiledTuboid(tuboid_dir)
            entries.append({
                'tuboid': tuboid,
                'label': self._taxonomy_mapper.level_dict_to_label(r.to_dict())
            })

        summary = {}
        for e in entries:
            lab = e['label']
            if lab not in summary:
                summary[lab] = 0
            summary[lab] += 1
        logging.info('Label summary in whole dataset (N=%i): %s', len(entries),
                     ["%s (%i) -> %i" % (self._taxonomy_mapper.label_to_pattern(k), k, summary[k])
                      for k in sorted(summary.keys())])
        return entries

    def visualise(self, subset: str = 'train', augment: bool = False, interactive: bool = True):
        raise NotImplementedError()

    def _get_torch_dataset(self, subset: str = 'train', augment: bool = False) -> torch.utils.data.Dataset:
        assert subset in {'train', 'val'}, 'subset should be either "train" or "val"'
        data = self._training_data if subset == 'train' else self._validation_data
        return OurTorchDataset(data, augment=augment)
