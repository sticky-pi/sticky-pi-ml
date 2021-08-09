import pandas as pd
import os
import dotenv
import argparse
import logging
import hashlib
import numpy as np
from shapely.geometry import Polygon
import logging
import cv2
from typing import List, Tuple, Union, IO
import datetime

STRING_DATETIME_FORMAT = '%Y-%m-%d_%H-%M-%S'


def md5(file: Union[IO, str], chunksize=32768):
    # if the file is a path, open and recurse
    if type(file) == str:
        with open(file, 'rb') as f:
            return md5(f)
    try:
        hash_md5 = hashlib.md5()
        for chunk in iter(lambda: file.read(chunksize), b""):
            hash_md5.update(chunk)
    finally:
        file.seek(0)
    return hash_md5.hexdigest()


def iou_match_pairs(arr: np.ndarray, iou_threshold: float) -> List[Tuple[int, int]]:
    """
    :param arr: a triangular 2d array containing iou values
    :param iou_threshold: the threshold under which two objects do not match
    :return: A list of matched object, by index. None for no match.
    """
    pairs = []
    arr[arr < iou_threshold] = 0

    gt_not_in_im = np.where(np.sum(arr, axis=1) == 0)[0]
    im_not_in_gt = np.where(np.sum(arr, axis=0) == 0)[0]

    for g in gt_not_in_im:
        pairs.append((g, None))

    for i in im_not_in_gt:
        pairs.append((None, i))

    while np.sum(arr) > 0:
        i, j = np.unravel_index(arr.argmax(), arr.shape)
        pairs.append((i, j))
        arr[i, :] = 0
        arr[:, j] = 0
    return pairs


def pad_to_square(array: np.ndarray, size: int):
    old_size = array.shape[:2]  # old_size is in (height, width) format
    ratio = float(size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format

    array = cv2.resize(array, (new_size[1], new_size[0]))

    delta_w = size - new_size[1]
    delta_h = size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(array, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)
    return new_im


def iou(poly1: Polygon, poly2: Polygon):

    try:
        inter = poly1.intersection(poly2).area
        if inter == 0:
            return 0
        return inter/poly1.union(poly2).area
    except Exception as e:
        logging.error(e)
        return 0


def detectron_to_pytorch_transform(Class):
    """
    Takes a transform class from detectron2 and return a regular pytorch transform class
    :param Class: a class inherited from detectron2.data.transform.Transform
    :return:
    """
    class MyClass(Class):
        def __call__(self, *args, **kwargs):
            return self.get_transform(*args, **kwargs).apply_image(*args, **kwargs).copy()
    return MyClass


class MLScriptParser(argparse.ArgumentParser):
    _valid_actions = {'fetch', 'train', 'qc', 'validate', 'push', 'predict', 'candidates'}
    _required_env_vars = ['BUNDLE_ROOT_DIR', 'LOCAL_CLIENT_DIR',
                          'API_HOST', 'API_USER', 'API_PASSWORD']

    def __init__(self, config_file=None):
        super().__init__()

        self.add_argument("action", help=str(self._valid_actions))

        self.add_argument("-v", "--verbose", dest="verbose", default=False,
                          help="verbose",
                          action="store_true")

        self.add_argument("-D", "--debug", dest="debug", default=False,
                          help="debug",
                          action="store_true")

        self.add_argument("-r", "--restart-training", dest="restart_training", default=False, action="store_true")
        self.add_argument("-g", "--gpu", dest="gpu", default=False, help="GPU", action="store_true")
        self.add_argument("-l", "--local-api", dest="local_api", default=False, help="Whether to use the local api", action="store_true")
        self._config_file = config_file

    def _get_env_conf(self):
        if self._config_file is not None:
            assert os.path.isfile(self._config_file)
            dotenv.load_dotenv(self._config_file)

        out = {}
        for var_name in self._required_env_vars:
            out[var_name] = os.getenv(var_name)
            if not out[var_name]:
                raise ValueError('No environment variable `%s''' % var_name)
        return out

    def get_opt_dict(self):
        args = self.parse_args()
        option_dict = vars(args)
        if option_dict['action'] not in self._valid_actions:
            logging.error('Wrong action!')
            self.print_help()
            exit(1)

        if option_dict['gpu']:
            option_dict['device'] = 'cuda'
        else:
            option_dict['device'] = 'cpu'

        if option_dict['verbose']:
            logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S',
                                level=logging.INFO)
        if option_dict['debug']:
            logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S',
                                level=logging.DEBUG)

        # 'BUNDLE_DIR', 'LOCAL_CLIENT_DIR'
        env_conf = self._get_env_conf()
        option_dict.update(env_conf)
        return option_dict


def string_to_datetime(string):
    return datetime.datetime.strptime(string, STRING_DATETIME_FORMAT)


def datetime_to_string(dt):
    if pd.isnull(dt):
        return None
    return datetime.datetime.strftime(dt, STRING_DATETIME_FORMAT)


