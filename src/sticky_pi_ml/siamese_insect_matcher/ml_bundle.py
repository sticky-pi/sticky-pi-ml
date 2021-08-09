from sticky_pi_ml.ml_bundle import BaseMLBundle
from sticky_pi_ml.siamese_insect_matcher.dataset import Dataset
import yaml


class MLBundle(BaseMLBundle):
    _name = 'siamese-insect-matcher'
    _DatasetClass = Dataset

    def _configure(self, config_file, device):
        with open(config_file, 'r') as file:
            config = yaml.load(file, Loader=yaml.SafeLoader)

        config['WEIGHTS'] = self._weight_file
        config['DEVICE'] = device
        return config

try:
    from sticky_pi_ml.ml_bundle import BaseClientMLBundle

    class ClientMLBundle(MLBundle, BaseClientMLBundle):
        pass
except ImportError:
    pass