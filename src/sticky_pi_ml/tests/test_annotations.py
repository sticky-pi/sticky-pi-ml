import unittest
from sticky_pi_ml.annotations import Annotation, DictAnnotation
import numpy as np


class TestAnnotations(unittest.TestCase):
    _test_image = "sticky_pi_dir/raw_images/7168f343/7168f343.2019-12-03_23-06-09.jpg"
    _image_shape = (1944, 2592, 3)

    def test_annotate(self):
        contour = np.array([[[1, 2],[3, 4],[5, 6]]]).transpose((1, 0, 2))
        a = Annotation(contour, '#ff0000')
        print(a.svg_element())


    # def test_subimage(self):
    #     contour = np.array([[[1, 10],[10, 1],[10, 10]]]).transpose((1, 0, 2))
    #     a = Annotation(contour, '#ff0000')
    #     o = a.subimage()
        # import cv2
        # approx_c, _ = cv2.findContours(o,cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        # self.assertEqual(np.sum(approx_c - contour), 0)

    def test_dict(self):
        contour = np.array([[[1, 10],[10, 1],[10, 10]]]).transpose((1, 0, 2))
        a = Annotation(contour, '#ff0000')
        json = a.to_dict()
        da = DictAnnotation(json)

        self.assertEqual(np.sum(np.abs(da.contour- a.contour)), 0.0)
