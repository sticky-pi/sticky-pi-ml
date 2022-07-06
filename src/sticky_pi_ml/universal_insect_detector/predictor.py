import requests
import os
import tempfile
import shutil
import math
import logging
import cv2
import numpy as np
import torch
from shapely.geometry import Polygon
from detectron2.engine import DefaultPredictor
from sticky_pi_ml.predictor import BasePredictor
from sticky_pi_ml.universal_insect_detector.ml_bundle import MLBundle

from sticky_pi_ml.universal_insect_detector.palette import Palette
from sticky_pi_ml.annotations import Annotation
from sticky_pi_ml.image import Image

import pandas as pd
from typing import Union

try:
    from sticky_pi_ml.universal_insect_detector.ml_bundle import ClientMLBundle
    from sticky_pi_api.types import InfoType

    Ml_bundle_type = Union[ClientMLBundle, MLBundle]
except ImportError:
    from typing import Any as InfoType

    logging.warning('Failed to load sticky_pi_api. Will not be able to use client MLBundles')
    Ml_bundle_type = MLBundle


class Predictor(BasePredictor):
    _detect_client_chunk_size = 64
    _minimum_tile_overlap = 500
    _score_threshold = 0.85
    _iou_threshold = 0.33

    def __init__(self, ml_bundle: Ml_bundle_type):
        super().__init__(ml_bundle)
        self._min_width = self._ml_bundle.config.MIN_MAX_OBJ_SIZE[0]
        self._palette = Palette({k: v for k, v in self._ml_bundle.config.CLASSES})
        self._detectron_predictor = DefaultPredictor(self._ml_bundle.config)

    def detect_client(self, info: InfoType = None, *args, **kwargs):
        assert issubclass(type(self._ml_bundle), ClientMLBundle), \
            "This method only works for MLBundles linked to a client"

        client = self._ml_bundle.client

        if info is None:
            info = [{'device': '%',
                     'start_datetime': "1970-01-01_00-00-00",
                     'end_datetime': "2070-01-01_00-00-00",
                     "algo_name": self.name,
                     "algo_version": self.version}]
            logging.info('No info provided. Fetching all annotations')

        logging.info(f"Info for `get_images_to_annotate`: {info}")
        while True:
            client_resp = client.get_images_to_annotate(info, what='image')

            if len(client_resp) == 0:
                return
            #
            # df = pd.DataFrame(client_resp)

            # query = [df.iloc[i][['device', 'datetime']].to_dict() for i in range(len(df))]
            # image_data = client.get_images(info=query, what='image')
            urls = [im['url'] for im in client_resp]

            all_annots = []
            for u in urls:
                temp_dir = None
                try:
                    if not os.path.isfile(u):
                        temp_dir = tempfile.mkdtemp()
                        filename = os.path.basename(u).split('?')[0]
                        resp = requests.get(u).content
                        with open(os.path.join(temp_dir, filename), 'wb') as file:
                            file.write(resp)
                        u = os.path.join(temp_dir, filename)

                    im = Image(u)
                    annotated_im = self.detect(im, *args, **kwargs)
                    logging.info('Detecting in image %s' % im)
                    annots = annotated_im.annotation_dict(as_json=False)
                    all_annots.append(annots)
                    logging.info("Staging annotations: %s" % annotated_im)
                finally:
                    if temp_dir:
                        shutil.rmtree(temp_dir)

            logging.info("Sending %i annotations to client" % len(all_annots))
            client.put_uid_annotations(all_annots)

    def detect(self, image: Image, *args, **kwargs) -> Image:

        instances = self._detect_instances(image, *args, **kwargs)
        new_image = image.copy()
        new_image.set_annotations(instances)
        new_image.tag_detector_version(self._name, self.version)
        return new_image

    def _mask_to_polygons(self, mask, offset, dilate_kernel_size=3):
        mask = np.ascontiguousarray(mask.cpu())  # some versions of cv2 does not support incontiguous arr
        kernel = np.ones((dilate_kernel_size, dilate_kernel_size))
        mask = cv2.dilate(mask.astype(np.uint8), kernel)
        contours, _ = cv2.findContours(mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       method=cv2.CHAIN_APPROX_SIMPLE,
                                       offset=offset)[-2:]  # for cv3 cv4 compat

        new_contours = []
        for c in contours:
            try:
                rect = cv2.minAreaRect(c)
                (_, _), (width, height), _ = rect

            except Exception as e:
                width, height = 0, 0
                logging.warning(e)

            if width < height:
                width, height = height, width

            if width > self._min_width / 2:
                new_contours.append(c)

        logging.debug("Size-excluded %i contours" % (len(contours) - len(new_contours)))
        contours, new_contours = new_contours, None
        out = []
        for c in contours:
            epsilon = 0.006 * cv2.arcLength(c, True)
            out.append(cv2.approxPolyDP(c, epsilon, True))
        contours = out
        if len(contours) == 0:
            return None
        largest_contour = np.argmax([cv2.contourArea(c) for c in contours])
        return contours[largest_contour]

    def _detect_instances(self, img: Image):
        polys = []
        classes = []
        logging.debug(img)

        array = cv2.copyMakeBorder(img.read(),
                                   self._ml_bundle.config.ORIGINAL_IMAGE_PADDING,
                                   self._ml_bundle.config.ORIGINAL_IMAGE_PADDING,
                                   self._ml_bundle.config.ORIGINAL_IMAGE_PADDING,
                                   self._ml_bundle.config.ORIGINAL_IMAGE_PADDING,
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # array = img.read()

        # todo
        # make exception to removing edge objects on the edge of the actual image
        # think about what to do when object fully overlap as they come from multiple detections
        # self intersecting contours :(

        if array.shape[1] <= 1024:
            x_range = [0]
            x_n_tiles = 1
        else:
            x_n_tiles = math.ceil(1 + (array.shape[1] - 1024) / (1024 - self._minimum_tile_overlap))
            x_stride = (array.shape[1] - 1024) // (x_n_tiles - 1)
            x_range = [r for r in range(0, array.shape[1] - 1023, x_stride)]

        if array.shape[0] <= 1024:
            y_range = [0]
            y_n_tiles = 1

        else:
            y_n_tiles = math.ceil(1 + (array.shape[0] - 1024) / (1024 - self._minimum_tile_overlap))
            y_stride = (array.shape[0] - 1024) // (y_n_tiles - 1)
            y_range = [r for r in range(0, array.shape[0] - 1023, y_stride)]
        offsets = []
        for n, j in enumerate(y_range):
            for m, i in enumerate(x_range):
                offsets.append(((m, n), (i, j)))

        for i, ((m, n), o) in enumerate(offsets):
            logging.info(f"{img.filename}, {i}/{len(offsets)}")
            im_1 = array[o[1]: (o[1] + 1024), o[0]: (o[0] + 1024)]
            p = self._detectron_predictor(im_1)
            p_bt = p['instances'].pred_boxes.tensor
            big_enough = torch.zeros_like(p_bt[:, 0], dtype=torch.bool)
            big_enough = big_enough.__or__(p_bt[:, 2] - p_bt[:, 0] > self._ml_bundle.config.MIN_MAX_OBJ_SIZE[0])
            big_enough = big_enough.__or__(p_bt[:, 3] - p_bt[:, 1] > self._ml_bundle.config.MIN_MAX_OBJ_SIZE[0])

            # print("Keeping, removing")
            # print(sum(big_enough), len(big_enough))
            # p['instances'] = p['instances'][big_enough]
            # print(len(p['instances']))

            # p_bt = p['instances'].pred_boxes.tensor

            # p['instances'] = p['instances'][]
            # we remove redundant edge instances as they should overlap
            non_edge_cases = torch.ones_like(p_bt[:, 0], dtype=torch.bool)

            if m > 0:
                non_edge_cases = non_edge_cases.__and__(p_bt[:, 0] > 32)
            if m < x_n_tiles - 1:
                non_edge_cases = non_edge_cases.__and__(p_bt[:, 2] < 1024 - 32)

            if n > 0:
                non_edge_cases = non_edge_cases.__and__(p_bt[:, 1] > 32)
            if n < y_n_tiles - 1:
                non_edge_cases = non_edge_cases.__and__(p_bt[:, 3] < 1024 - 32)


            p['instances'] = p['instances'][non_edge_cases.__and__(big_enough)]
            p['instances'] = p['instances'][p['instances'].scores > self._score_threshold]
            classes_for_one_inst = []
            poly_for_one_inst = []

            #fixme, this could be parallelised
            for i in range(len(p['instances'])):
                instance_offset = (o[0] - self._ml_bundle.config.ORIGINAL_IMAGE_PADDING,
                                   o[1] - self._ml_bundle.config.ORIGINAL_IMAGE_PADDING)

                poly = self._mask_to_polygons(p['instances'].pred_masks[i, :, :],
                                              offset=instance_offset)
                if poly is not None:
                    poly_for_one_inst.append(poly)
                    classes_for_one_inst.append(
                        int(p['instances'].pred_classes[i]) + 1)  # as we want one-indexed classes
            polys.append(poly_for_one_inst)
            classes.append(classes_for_one_inst)

        overlappers = []
        for i in range(len(offsets)):
            overlappers_sub = []
            for j in range(len(offsets)):
                if i != j and abs(offsets[j][1][0] - offsets[i][1][0]) < 1024 and abs(
                        offsets[j][1][1] - offsets[i][1][1]) < 1024:
                    overlappers_sub.append(j)
            overlappers.append(overlappers_sub)

        all_valid = []  # (origin, pred_class, poly)

        # merge predictions from several overlaping tiles
        for origin, poly_one_pred in enumerate(polys):
            for i, p1 in enumerate(poly_one_pred):
                add = True
                p_shape_1 = Polygon(np.squeeze(p1))
                for v in all_valid:
                    if origin not in overlappers[v[0]]:
                        continue
                    p2 = v[2]
                    p_shape_2 = Polygon(np.squeeze(p2))

                    # todo check bounding box overlap
                    try:
                        iou = p_shape_1.intersection(p_shape_2).area / p_shape_1.union(p_shape_2).area
                    except Exception as e:
                        iou = 1  # fixme topological exception
                    if iou > self._iou_threshold:
                        add = False
                        continue
                if add:
                    all_valid.append((origin, classes[origin][i], p1))
        annotation_list = []
        for _, pred_class, poly in all_valid:
            stroke = self._palette.get_stroke_from_id(pred_class)
            class_name = self._palette.get_class_from_id(pred_class)
            a = Annotation(poly, parent_image=img, stroke_colour=stroke, name=class_name)
            annotation_list.append(a)
        return annotation_list
