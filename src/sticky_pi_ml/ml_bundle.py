import os
from sticky_pi_ml.dataset import BaseDataset
from sticky_pi_ml.utils import md5
import logging
from abc import ABC


class BaseMLBundle(ABC):
    _data_dirname = 'data'      #
    _config_dirname = 'config'  #
    _config_filename = 'config.yaml'
    _output_dirname = 'output'   #
    _model_filename = 'model_final.pth'
    _version_filename = '.version.txt'
    _name = None
    _DatasetClass = None  # must be implemented

    def __init__(self, root_dir: str, device: str = 'cpu', cache_dir=None):
        """
        An abstract class that organises all the components of a ML project:

        * training/validation data files -- in ``/data``
        * configuration files -- in ``/config``
        * weight files (i.e. pretrained or resulting of training)  -- in ``/output``

        All components are stored in ``root_dir`` and the class provided utilities to parse inputs, generate ``torch``
        datasets, synchronise the data to an API...
        :param root_dir: the location of the files
        """
        self._cache_dir = cache_dir
        self._root_dir = root_dir
        self._device = device

        if not os.path.isdir(root_dir):
            logging.warning("%s is not a directory, creating it" % root_dir)
            # assert os.path.dirname(os.path.normpath(root_dir)),
            os.mkdir(root_dir)

        self._output_dir = os.path.join(self._root_dir, self._output_dirname)
        self._config_dir = os.path.join(self._root_dir, self._config_dirname)
        self._data_dir = os.path.join(self._root_dir, self._data_dirname)

        if self._cache_dir is None:
            import tempfile
            import atexit
            import shutil
            self._cache_dir = tempfile.mkdtemp(prefix='sticky_pi_%s_' % self._name)
            atexit.register(shutil.rmtree, self._cache_dir)

            os.makedirs(self._output_dir, exist_ok=True)

        config_file = os.path.join(self._config_dir, self._config_filename)
        self._weight_file = os.path.join(self._output_dir, self._model_filename)

        if not os.path.isdir(self._data_dir):
            logging.warning("Data dir does not exist. making it: %s" % self._data_dir)
            os.makedirs(self._data_dir, exist_ok=True)

        if not os.path.isdir(self._config_dir):
            logging.warning("config dir does not exist. making it: %s" % self._config_dir)
            os.makedirs(self._config_dir, exist_ok=True)

        if not os.path.isdir(self._output_dir):
            logging.warning("Model dir does not exist. making it: %s" % self._output_dir)
            os.makedirs(self._output_dir, exist_ok=True)

        if not os.path.isfile(config_file):
            logging.warning("Configuration file %s is does not exist (yet?)" % config_file)
            self._config = None
            self._dataset = None
        else:
            self._config = self._configure(config_file, self._device)
            self._dataset = self._DatasetClass(self._data_dir, self._config, self._cache_dir)

    def _configure(self, config_file, device):
        raise NotImplementedError

    @property
    def dataset(self) -> BaseDataset:
        return self._dataset

    @property
    def name(self) -> str:
        return self._name

    @property
    def config(self) -> dict:
        return self._config

    @property
    def version(self):
        file = self._weight_file
        m = md5(file)
        version_file = os.path.join(self._output_dir, self._version_filename)

        if not os.path.isfile(version_file):
            self._tag_version(file, m)

        with open(version_file, 'r') as f:
            t, md5sum = f.read().rstrip().split('-')
        t = int(t)

        if m != md5sum:
            return self._tag_version(file, m)
        else:
            mtime = t
            return "%i-%s" % (mtime, m)

    def _tag_version(self, file, md5sum):
        mtime = os.path.getmtime(file)
        version = "%i-%s" % (mtime, md5sum)
        with open(os.path.join(self._output_dir, self._version_filename), 'w') as f:
            f.write(version)
            logging.info('Local version md5 different from version file. Tagging new version: "%i-%s"' % (mtime, md5sum))
        return version
    @property
    def weight_file(self):
        return self._weight_file

try:
    from sticky_pi_api.client import BaseClient
    class BaseClientMLBundle(BaseMLBundle, ABC):
        def __init__(self, root_dir: str, client: BaseClient, device: str = 'cpu', cache_dir=None):
            super().__init__(root_dir, device, cache_dir)
            self._client = client

        @property
        def client(self):
            return self._client

        def sync_local_to_remote(self, what: str = 'all'):
            assert what in {'all', 'data', 'model'}
            # we trigger version tagging
            try:
                _ = self.version
            except FileNotFoundError as e:
                logging.warning(e)
            self._client.put_ml_bundle_dir(self._name, self._root_dir, what)

        def sync_remote_to_local(self, what: str = 'all'):
            assert what in {'all', 'data', 'model'}
            self._client.get_ml_bundle_dir(self._name, self._root_dir, what)
            self.__init__(self._root_dir, self._client, self._device, self._cache_dir)

except ImportError:
    logging.warning('Failed to load sticky_pi_api. Will not be able to use client MLBundles')
