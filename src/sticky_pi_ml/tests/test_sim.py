import glob
import os
import logging
import shutil
import tempfile
import numpy as np
import unittest
from sticky_pi_ml.siamese_insect_matcher.ml_bundle import MLBundle, ClientMLBundle
from sticky_pi_ml.siamese_insect_matcher.trainer import Trainer
from sticky_pi_ml.siamese_insect_matcher.predictor import Predictor
from sticky_pi_ml.siamese_insect_matcher.candidates import make_candidates
from sticky_pi_ml.image import SVGImage
from sticky_pi_api.client import LocalClient

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)


test_dir = os.path.dirname(__file__)


class MockCNN(object):
    def __call__(self, *args, **kwargs):
        return 1.0

    def eval(self):
        pass

    def load_state_dict(self, file):
        pass


class MockPredictor(Predictor):
    _model_class = MockCNN


class TestSIM(unittest.TestCase):
    _bundle_dir = os.path.join(test_dir, 'ml_bundles/siamese-insect-matcher')

    # _test_image = os.path.join(test_dir,
    #                            "ml_bundles/siamese-insect-matcher/data",
    #                            "0a5bb6f4.2020-06-24_22-03-58.2020-06-24_22-22-42.svg")

    _test_images = [i for i in sorted(glob.glob(os.path.join(test_dir, "raw_images/**/*.jpg")))]
    _raw_images_dir = os.path.join(test_dir, "raw_images")
    _test_reg_images = [SVGImage(i) for i in sorted(glob.glob(os.path.join(test_dir,
                                                                           "ml_bundles/universal-insect-detector/data",
                                                                           "0a5bb6f4*.svg")))]


    def test_trainer(self):
        #fixme here, the bundle dir should be copied to avoid modifications?
        bndl = MLBundle(self._bundle_dir)
        t = Trainer(bndl)
        t.resume_or_load(resume=True)
        t.train()

    # def test_Predictor(self):
    #     bndl = MLBundle(self._bundle_dir)
    #     pred = MockPredictor(bndl)
    #     pred = Predictor(bndl)
    #
    #     # pred.match_two_annots()
    #     o = pred.match_two_images(*self._test_reg_images[1:3])

    # def test_make_candidate(self):
    #     from sticky_pi_ml.tests.test_uid import MockPredictor as MockUIDPredictor
    #     from sticky_pi_ml.universal_insect_detector.ml_bundle import ClientMLBundle as ClientUIDMLBundle
    #
    #     client_temp_dir = tempfile.mkdtemp(prefix='sticky_pi_client_')
    #     temp_dst_bundle = tempfile.mkdtemp(prefix='sticky_pi_test_')
    #     try:
    #         uid_bundle_dir = os.path.join(test_dir, 'ml_bundles/universal-insect-detector')
    #         cli = LocalClient(client_temp_dir)
    #         bndl = ClientUIDMLBundle(uid_bundle_dir, cli)
    #         bndl.sync_local_to_remote()
    #         ims_to_pred = [im for im in sorted(glob.glob(os.path.join(self._raw_images_dir, '**', '*.jpg')))]
    #         cli.put_images(ims_to_pred)
    #         pred = MockUIDPredictor(bndl)
    #         pred.detect_client()
    #         make_candidates(cli, out_dir=temp_dst_bundle)
    #
    #     finally:
    #         shutil.rmtree(client_temp_dir)
    #         shutil.rmtree(temp_dst_bundle)
    # #
