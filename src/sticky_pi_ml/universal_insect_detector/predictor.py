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
from sticky_pi_ml.universal_insect_detector.ml_bundle import MLBundle, ClientMLBundle
from sticky_pi_ml.universal_insect_detector.palette import Palette
from sticky_pi_ml.annotations import Annotation
from sticky_pi_ml.image import Image
from sticky_pi_api.client import BaseClient
from sticky_pi_api.types import InfoType
import pandas as pd
from typing import Union



class Predictor(BasePredictor):

    _detect_client_chunk_size = 64
    _minimum_tile_overlap = 500

    def __init__(self, ml_bundle: Union[ClientMLBundle, MLBundle]):
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
                     'end_datetime': "2070-01-01_00-00-00"}]
            logging.info('No info provided. Fetching all annotations')
        while True:
            client_resp = client.get_images_with_uid_annotations_series(info, what_image='metadata', what_annotation='metadata')

            if len(client_resp) == 0:
                return

            df = pd.DataFrame(client_resp)


            if 'algo_name' not in df.columns:
                logging.info('No annotations for the requested images. Fetching all!')
                df['algo_version'] = None
                df['algo_name'] = ""

            df = df.sort_values(by=['algo_version', 'datetime'])
            df = df.drop_duplicates(subset=['id'], keep='last')

            # here, we filter/sort df to keep only images that are not annotated by this version.
            # we sort by version tag

            conditions = (self.version > df.algo_version) | \
                         (df.algo_version.isnull()) | \
                         (self.name != df.algo_name)

            df = df[conditions]
            if len(df) == 0:
                logging.info('All annotations uploaded!')
                return

            query = [df.iloc[i][['device', 'datetime']].to_dict() for i in range(min(len(df), self._detect_client_chunk_size))]
            image_data = client.get_images(info=query, what='image')
            urls = [im['url'] for im in image_data]

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

    def detect(self, image, *args, **kwargs) -> Image:
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
                                       offset=offset)[-2:] # for cv3 cv4 compat

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

    def _detect_instances(self, img, score_threshold=.50):
        polys = []
        classes = []
        logging.debug(img)
        array = img.read()

        # todo exclude object based on size
        # make exception to removing edge objects on the edge of the actual image
        # think about what to do when object fully overlap as they come from multiple detections
        # self intersecting contours :(


        x_n_tiles = math.ceil(1 + (array.shape[1] - 1024) / (1024 - self._minimum_tile_overlap))
        x_stride = (array.shape[1] - 1024) // (x_n_tiles - 1)
        y_n_tiles = math.ceil(1 + (array.shape[0] - 1024) / (1024 - self._minimum_tile_overlap))
        y_stride = (array.shape[0] - 1024) // (y_n_tiles - 1)

        offsets = []
        for n, j in enumerate(range(0, array.shape[0] - 1023, y_stride)):
            for m, i in enumerate(range(0, array.shape[1] - 1023, x_stride)):
                offsets.append(((m, n), (i, j)))

        for ((m, n), o) in offsets:
            im_1 = array[o[1]: (o[1] + 1024), o[0]: (o[0] + 1024)]
            p = self._detectron_predictor(im_1)
            p_bt = p['instances'].pred_boxes.tensor
            non_edge_cases = torch.ones_like(p_bt[:, 0], dtype=torch.bool)
            if m > 0:
                non_edge_cases = non_edge_cases.__and__(p_bt[:, 0] > 32)
            if m < x_n_tiles - 1:
                non_edge_cases = non_edge_cases.__and__(p_bt[:, 2] < 1024 - 32)

            if n > 0:
                non_edge_cases = non_edge_cases.__and__(p_bt[:, 1] > 32)
            if n < y_n_tiles - 1:
                non_edge_cases = non_edge_cases.__and__(p_bt[:, 3] < 1024 - 32)

            p['instances'] = p['instances'][non_edge_cases]
            p['instances'] = p['instances'][p['instances'].scores > score_threshold]
            classes_for_one_inst = []
            poly_for_one_inst = []
            for i in range(len(p['instances'])):
                poly = self._mask_to_polygons(p['instances'].pred_masks[i, :, :], offset=o)
                if poly is not None:
                    poly_for_one_inst.append(poly)
                    classes_for_one_inst.append(int(p['instances'].pred_classes[i]) + 1) # as we want one-indexed classes
            polys.append(poly_for_one_inst)
            classes.append(classes_for_one_inst)

        overlappers = []
        for i in range(len(offsets)):
            overlappers_sub = []
            for j in range(len(offsets)):
                if i != j and abs(offsets[j][1][0] - offsets[i][1][0]) < 1024 and abs(offsets[j][1][1] - offsets[i][1][1]) < 1024:
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
                    if iou > 0.5:
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

