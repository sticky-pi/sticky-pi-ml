import unittest
from sticky_pi_ml.image import Image, SVGImage, ImageJsonAnnotations, ImageSeries
import os
import logging
import pytz
import datetime
import numpy as np
import tempfile
from sticky_pi_ml.annotations import Annotation
import glob
import shutil

test_dir = os.path.dirname(__file__)

class TestImage(unittest.TestCase):
    # _test_image = "raw_images/1b74105a/1b74105a"
    _raw_images_dir = os.path.join(test_dir, "raw_images")
    _bundle_dir = os.path.join(test_dir, 'ml_bundles/universal-insect-detector')
    _test_image = os.path.join(test_dir, "raw_images/1b74105a/1b74105a.2020-07-05_10-07-16.jpg")
    # _test_svg_images = ("sticky_pi_dir/raw_images/01abcabc.2020-01-01_01-02-03.svg",
    #                     "sticky_pi_dir/raw_images//d59ff54a.2020-03-03_18-04-20.svg",
    #                     "sticky_pi_dir/raw_images/abcdef00.2021-01-01_00-00-01.svg")
    #
    # _test_legacy_svg_image = "sticky_pi_dir/raw_images/legacy_3b8bb733.2019-12-10_15-46-49.svg"
    # _test_image_nonext = "sticky_pi_dir/raw_images/7168f343/7168f3a3.2019-12-03_23-06-09.jpg"
    # _test_combine_path_svg_image = "sticky_pi_dir/raw_images/combined-path_afbd6f68.2020-04-23_20-49-21.svg"
    _image_shape = (1944, 2592, 3)
    # maxDiff = None

    def test_init(self):
        timezone = pytz.timezone("UTC")
        full_path = os.path.join(os.path.dirname(__file__), self._test_image)
        im = Image(full_path)
        print(im)
        self.assertEqual(im.datetime, datetime.datetime(2020, 7, 5, 10, 7, 16))
        self.assertEqual(im.device, '1b74105a')
        self.assertEqual(im.filename, '1b74105a.2020-07-05_10-07-16.jpg')
        self.assertEqual(im.path, full_path)

    def test_image_series(self):
        from sticky_pi_ml.universal_insect_detector.ml_bundle import ClientMLBundle
        from sticky_pi_api.client import LocalClient
        from sticky_pi_ml.tests.test_uid import MockPredictor

        client_temp_dir = tempfile.mkdtemp(prefix='sticky_pi_client_')
        # the di dirname is used to identify the ML bundle
        todel = tempfile.mkdtemp(prefix='sticky_pi_test_')
        try:
            temp_dst_bundle = os.path.join(todel, 'universal-insect-detector')
            os.makedirs(temp_dst_bundle)
            cli = LocalClient(client_temp_dir)

            bndl = ClientMLBundle(self._bundle_dir, cli)
            bndl.sync_local_to_remote()
            ims_to_pred = [im for im in sorted(glob.glob(os.path.join(self._raw_images_dir, '**', '*.jpg')))]
            cli.put_images(ims_to_pred)

            pred = MockPredictor(bndl)

            series = ImageSeries(device='0a5bb6f4',
                                 start_datetime='2020-01-01_00-00-00',
                                 end_datetime='2021-01-01_00-00-00')

            series.populate_from_client(cli)
            self.assertEqual(len(series), 0)
            pred.detect_client()
            series.populate_from_client(cli)
            self.assertEqual(len(series), 5)
            # should populate only with the last version of the algorithm available
            pred._version = '1604062778-262624ad1767b977801645a8addefbe6'
            pred.detect_client()
            series.populate_from_client(cli)
            self.assertEqual(len(series), 5)
            for s in series:
                self.assertTrue(s.algo_version == pred._version)

        finally:
            shutil.rmtree(client_temp_dir)
            shutil.rmtree(todel)
            pass

    def test_read(self):
        full_path = os.path.join(os.path.dirname(__file__), self._test_image)
        im = Image(full_path)
        array = im.read()
        self.assertEqual(array.shape, self._image_shape)

    def test_to_svg(self):
        self._to_svg(self._test_image)


    # def test_json_image(self):
    #     im = self._test_svg_images[2]
    #     svg_ori = SVGImage(im)
    #     json_annots = svg_ori.json_annotations()
    #     svg_ori.tag_detector_version('hello', 'world')
    #     json_annots = svg_ori.json_annotations()
    #
    #     # ImageJsonAnnotations(json_annots)


    # def test_svg_to_svg(self):
    #     im = self._test_svg_images[2]
    #     svg_ori = SVGImage(im)
    #     array = svg_ori.read()
    #
    #     tmp_dir = tempfile.mkdtemp(prefix='sticky_pi_test_')
    #     try:
    #
    #         target = os.path.join(tmp_dir, '01abcabc.2020-01-01_01-02-03.svg')
    #         svg_ori.to_svg(target)
    #         svg_new = SVGImage(target)
    #         array_new = svg_new.read()
    #         self.assertEqual(np.sum(array - array_new), 0)
    #     finally:
    #         shutil.rmtree(tmp_dir)
    #         pass
    #
    def _to_svg(self, img):

        full_path = os.path.join(os.path.dirname(__file__), img)
        im = Image(full_path)

        contours = [np.array([[[551, 552],[883, 884],[995, 596]]]).transpose((1, 0, 2)),
                    np.array([[[51, 52], [93, 94], [500, 6]]]).transpose((1, 0, 2)),
                    np.array([[[101, 102], [203, 204], [405, 306]]]).transpose((1, 0, 2))]
        an = [Annotation(c, '#ffff00') for c in contours ]

        im.set_annotations(an)

        tmp_dir = tempfile.mkdtemp(prefix='sticky_pi_test_')
        try:

            target = os.path.join(tmp_dir, '01abcabc.2020-01-01_01-02-03.svg')

            im.to_svg(target)
            imsvg = SVGImage(target)

            self.assertEqual(im.shape, imsvg.shape)

            self.assertDictEqual(im.metadata, imsvg.metadata)
            svg_contour = [a._contour for a in imsvg._annotations]
            for c,s in zip(contours, svg_contour):
                self.assertTrue(np.sum(c-s) == 0)

            self.assertTrue(os.path.exists(target))


        finally:
            # logging.warning(tmp_dir)
            shutil.rmtree(tmp_dir)
            pass

    #
    # def test_extract_jpeg(self):
    #     for im in self._test_svg_images:
    #         full_path = os.path.join(os.path.dirname(__file__), im)
    #         svgim = SVGImage(full_path)
    #
    #         tmp_dir = tempfile.mkdtemp(prefix='sticky_pi_test_')
    #         try:
    #             target = os.path.join(tmp_dir, '01abcabc.2020-01-01_01-02-03.jpg')
    #             svgim.extract_jpeg(target)
    #             im = Image(target)
    #             self.assertTrue(np.sum(im.read() - svgim.read()) == 0)
    #
    #         finally:
    #             shutil.rmtree(tmp_dir)
    #             pass
    #
    # def test_to_png(self):
    #
    #     full_path = os.path.join(os.path.dirname(__file__), self._test_image)
    #     im = Image(full_path)
    #
    #     contours = [np.array([[[551, 552],[883, 884],[995, 596]]]).transpose((1, 0, 2)),
    #                np.array([[[51, 52], [93, 94], [500, 6]]]).transpose((1, 0, 2)),
    #                np.array([[[101, 102], [203, 204], [405, 306]]]).transpose((1, 0, 2))]
    #     an = [Annotation(c, '#ffff00') for c in contours ]
    #
    #     im.set_annotations(an)
    #
    #     tmp_dir = tempfile.mkdtemp(prefix='sticky_pi_test_')
    #     try:
    #
    #         target = os.path.join(tmp_dir, '01abcabc.2020-01-01_01-02-03.png')
    #         im.to_png(target)
    #
    #     finally:
    #         import shutil
    #         shutil.rmtree(tmp_dir)
    #
    #
    # def test_legacy_svg(self):
    #
    #     full_path = os.path.join(os.path.dirname(__file__), self._test_legacy_svg_image)
    #     # todo expects warning here "Cannot extract metadata"
    #     svgim = SVGImage(full_path)
    #
    #     tmp_dir = tempfile.mkdtemp(prefix='sticky_pi_test_')
    #     try:
    #
    #         target = os.path.join(tmp_dir, '01abcabc.2020-01-01_01-02-03.jpg')
    #         svgim.extract_jpeg(target)
    #         im = Image(target)
    #         self.assertTrue(np.sum(im.read() - svgim.read()) == 0)
    #
    #     finally:
    #         shutil.rmtree(tmp_dir)
    #         pass
    #
    #
    # def test_combined_svg_path(self):
    #
    #     full_path = os.path.join(os.path.dirname(__file__), self._test_combine_path_svg_image)
    #     # todo expects warning here "Cannot extract metadata"
    #     svgim = SVGImage(full_path)




        #
        # tmp_dir = tempfile.mkdtemp(prefix='sticky_pi_test_')
        # try:
        #
        #     target = os.path.join(tmp_dir, '01abcabc.2020-01-01_01-02-03.jpg')
        #     svgim.extract_jpeg(target)
        #     im = Image(target)
        #     self.assertTrue(np.sum(im.read() - svgim.read()) == 0)
        #
        # finally:
        #     shutil.rmtree(tmp_dir)
        #     pass
