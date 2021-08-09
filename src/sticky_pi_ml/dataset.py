import shutil
import tempfile
import torch
from torch.utils.data import Dataset


class BaseDataset(object):
    _sub_datasets = {'train', 'val'}
    _md5_max_training = 'bf'  # use the md5 of object to allocate to train vs val (md5 > _md5_max_training <=> val)

    def __init__(self, data_dir: str, config, cache_dir: str):
        self._data_dir = data_dir
        self._config = config
        self._cache_dir = cache_dir
        self._is_prepared = False
        self._training_data = []
        self._validation_data = []

    def _prepare(self):
        raise NotImplementedError()

    def prepare(self):
        self._prepare()
        assert len(self._training_data) > 0, "Should have at least 1 images in Training set"
        assert len(self._validation_data) > 0, "Should have at least 1 images in Validation set"
        self._is_prepared = True

    def get_torch_data_loader(self, subset: str = 'train', shuffle: bool = True) -> torch.utils.data.DataLoader:
        assert subset in {'train', 'val'}, 'subset should be either "train" or "val"'
        augment = subset == 'train'
        to_load = self._get_torch_dataset(subset, augment=augment)

        out = torch.utils.data.DataLoader(to_load,
                                          batch_size=self._config['IMS_PER_BATCH'],
                                          shuffle=shuffle,
                                          num_workers=self._config['N_WORKERS'])
        return out

    def _get_torch_dataset(self, subset: str, augment: bool) -> torch.utils.data.Dataset:
        raise NotImplementedError()

    def visualise(self, subset: str = 'train', augment: bool = False, interactive: bool = True):
        raise NotImplementedError()

    @property
    def validation_data(self):
        return self._validation_data
    @property
    def training_data(self):
        return self._training_data
