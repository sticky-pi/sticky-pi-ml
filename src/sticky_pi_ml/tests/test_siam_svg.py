import unittest
from sticky_pi_ml.siamese_insect_matcher.siam_svg import SiamSVG, SVGImage
import os
import logging
import tempfile
import glob
import shutil

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

test_dir = os.path.dirname(__file__)


class TestSiamSVG(unittest.TestCase):

    _test_image = os.path.join(test_dir,
                               "ml_bundles/siamese-insect-matcher/data",
                               "0a5bb6f4.2020-06-24_22-03-58.2020-06-24_22-22-42.svg")
    _test_reg_images = [SVGImage(i) for i in sorted(glob.glob(os.path.join(test_dir,
                                              "ml_bundles/universal-insect-detector/data", "0a5bb6f4*.svg")))]


    def test_init(self):
        svg_img = SiamSVG(self._test_image)

        self.assertEqual(svg_img.device, '0a5bb6f4')

    def test_file_ops(self):
        temp_dir = tempfile.mkdtemp(prefix='sticky_pi_')
        try:
            hand_made_siam = SiamSVG.merge_two_images(self._test_reg_images[0],
                                                      self._test_reg_images[1],
                                                      dest_dir=temp_dir,
                                                      prematch=True)

            reread_siam = SiamSVG(hand_made_siam)

            self.assertEqual(reread_siam.extract_jpeg(as_buffer=True, id=0).read(),
                             self._test_reg_images[0].extract_jpeg(as_buffer=True).read())
            self.assertEqual(reread_siam.extract_jpeg(as_buffer=True, id=1).read(),
                             self._test_reg_images[1].extract_jpeg(as_buffer=True).read())
        finally:
            shutil.rmtree(temp_dir)

