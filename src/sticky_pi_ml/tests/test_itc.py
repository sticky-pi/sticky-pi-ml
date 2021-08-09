import glob
import os
import logging
import shutil
import tempfile
import numpy as np
import unittest
from sticky_pi_ml.insect_tuboid_classifier.ml_bundle import MLBundle, ClientMLBundle
from sticky_pi_ml.insect_tuboid_classifier.trainer import Trainer
from sticky_pi_ml.insect_tuboid_classifier.predictor import Predictor
from sticky_pi_api.client import LocalClient

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
import torch
class MockCNN(object):
    def __call__(self, *args, **kwargs):
        return torch.Tensor([1.0, 9, 9])

    def eval(self):
        pass

    def load_state_dict(self, file):
        pass


class MockPredictor(Predictor):
    def _make_net(self):
        return MockCNN()



test_dir = os.path.dirname(__file__)

class TestITC(unittest.TestCase):
    _bundle_dir = os.path.join(test_dir, 'ml_bundles/insect-tuboid-classifier')

    # _test_images = [i for i in sorted(glob.glob(os.path.join(test_dir, "raw_images/**/*.jpg")))]
    # _raw_images_dir = os.path.join(test_dir, "raw_images")
    # _test_reg_images = [SVGImage(i) for i in sorted(glob.glob(os.path.join(test_dir,
    #                                                                        "ml_bundles/universal-insect-detector/data",
    #                                                                        "0a5bb6f4*.svg")))]
    _tiled_tuboid_dir = os.path.join(test_dir, "tiled_tuboids")

    # def test_trainer(self):
    #     #fixme, here we should copy the bundle files to avoid corrupting them
    #     bndl = MLBundle(self._bundle_dir)
    #     t = Trainer(bndl)
    #     t.resume_or_load(resume=False)
    #     t.train()

    # def test_client_ml_bundle(self):
    #     client_temp_dir = tempfile.mkdtemp(prefix='sticky_pi_client_')
    #     todel = tempfile.mkdtemp(prefix='sticky_pi_test_')
    #
    #     try:
    #         temp_dst_bundle = os.path.join(todel, 'universal-insect-detector')
    #         os.makedirs(temp_dst_bundle)
    #         cli = LocalClient(client_temp_dir)
    #         bndl = ClientMLBundle(self._bundle_dir, cli)
    #         bndl.sync_remote_to_local()
    #
    #
    #         # should reinit the bundle after dl the data to the local dir
    #         bndl2.sync_remote_to_local()
    #         # bndl2 = ClientMLBundle(bndl2._root_dir, cli)
    #
    #         cf1 = bndl._config
    #         cf2 = bndl2._config
    #         cf1.MODEL.WEIGHTS = None
    #         cf2.MODEL.WEIGHTS = None
    #         cf1.OUTPUT_DIR = None
    #         cf2.OUTPUT_DIR = None
    #
    #         self.assertDictEqual(cf1, cf2)
    #         self.assertEqual([os.path.basename(p) for p in sorted(glob.glob(os.path.join(bndl.dataset._data_dir)))],
    #                          [os.path.basename(p) for p in sorted(glob.glob(os.path.join(bndl2.dataset._data_dir)))])
    #
    #     finally:
    #         shutil.rmtree(client_temp_dir)
    #         shutil.rmtree(todel)
    #         pass
    def test_client_predict(self):
        client_temp_dir = tempfile.mkdtemp(prefix='sticky_pi_client_')
        todel = tempfile.mkdtemp(prefix='sticky_pi_test_')
        try:
            temp_dst_bundle = os.path.join(todel, 'insect-tuboid-classifier')
            os.makedirs(temp_dst_bundle)
            cli = LocalClient(client_temp_dir)

            bndl = ClientMLBundle(self._bundle_dir, cli)
            bndl.sync_local_to_remote()

            tiled_tub_dirs = [os.path.dirname(im) for im in sorted(glob.glob(os.path.join(self._tiled_tuboid_dir, '**','**', 'tuboid.jpg')))]
            cli.put_tiled_tuboids(tiled_tub_dirs,
                                  series_info={'algo_name':'abcd',
                                               'algo_version': 'abcde',
                                               'device': '08038ade',
                                               'start_datetime': '2020-01-01_00-00-00',
                                               'end_datetime': '2021-01-01_00-00-00',
                                               'n_images':11,
                                               'n_tuboids':6},

                                  )

            pred = MockPredictor(bndl)
            pred._version = '1604062778-262624ad1767b977801645a8addefbe6'
            pred.predict_client('%', '2020-01-01_00-00-00', '2021-01-01_00-00-00')
            pred.predict_client('08038ade', '2020-01-01_00-00-00', '2021-01-01_00-00-00')

        finally:
            shutil.rmtree(client_temp_dir)
            shutil.rmtree(todel)
