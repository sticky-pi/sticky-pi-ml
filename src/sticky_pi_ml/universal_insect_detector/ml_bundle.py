import os
from sticky_pi_ml.ml_bundle import BaseMLBundle
from sticky_pi_ml.universal_insect_detector.dataset import Dataset
import yaml
from detectron2.config import get_cfg, CfgNode
from detectron2 import model_zoo


class MLBundle(BaseMLBundle):
    _name = 'universal-insect-detector'
    _DatasetClass = Dataset

    def _configure(self, config_file, device):

        config = get_cfg()

        # this is a hack to merge our own config variables into the detectron config
        with open(config_file, 'r') as file:
            config_dict = yaml.load(file,  Loader=yaml.SafeLoader)
            # The set of keys that are ONLY in out custom configuration
            custom_keys = set(config_dict.keys()).difference(config.keys())

            for k in custom_keys:
                if not k.startswith('_'):  # that avoid '_BASE_' to be parsed
                    setattr(config, k, config_dict[k])

        config.merge_from_file(config_file)
        config.DATASETS.TEST = (self._name + '_val',)
        config.DATASETS.TRAIN = (self._name + '_train',)
        config.MODEL.ROI_HEADS.NUM_CLASSES = len(config.CLASSES)
        config.OUTPUT_DIR = self._output_dir  # full path

        # # todo check this actually works when 1. training from scratch and 2. resuming
        # # also what happens if training is interrupted -- say by the HPC resource scheduler
        if os.path.isfile(self._weight_file):
            config.MODEL.WEIGHTS = self._weight_file
        else:
            config.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                os.path.join(config.BASE_MODEL_PREFIX, config_dict['_BASE_'])
            )

        config.MODEL.DEVICE = device
        return config


try:
    from sticky_pi_ml.ml_bundle import BaseClientMLBundle

    class ClientMLBundle(MLBundle, BaseClientMLBundle):
        pass
except ImportError:
    pass