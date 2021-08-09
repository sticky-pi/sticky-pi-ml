import numpy as np
import torch
import logging
from functools import lru_cache
from typing import Dict, Tuple, List, Any

from sticky_pi_api.types import InfoType

from sticky_pi_ml.predictor import BasePredictor
from sticky_pi_ml.siamese_insect_matcher.model import SiameseNet
from sticky_pi_ml.image import Image
from sticky_pi_ml.siamese_insect_matcher.ml_bundle import MLBundle, ClientMLBundle
from sticky_pi_ml.annotations import Annotation

from sticky_pi_ml.siamese_insect_matcher.dataset import Dataset, DataEntry


class Predictor(BasePredictor):

    _model_class = SiameseNet

    def __init__(self, ml_bundle: MLBundle):
        super().__init__(ml_bundle)
        self._max_delta_t = ml_bundle.config['MAX_DELTA_T_TO_MATCH']
        self._net = self._model_class()
        weights = self._ml_bundle.weight_file
        self._net.load_state_dict(torch.load(weights))
        self._net.eval()

    def match_two_images(self, im0: Image, im1: Image) -> Tuple[List[Tuple[str, str, float]], Dict[str, Any]]:
        an0 = im0.annotations
        an1 = im1.annotations
        nodes = {}
        for m, a0 in enumerate(an0):
            node_0 = '%s|%i' % (a0.parent_image.filename, m)
            nodes[node_0] = a0

        for n, a1 in enumerate(an1):
            node_1 = '%s|%i' % (a1.parent_image.filename, n)
            nodes[node_1] = a1

        arr = np.zeros((len(an0), len(an1)), dtype=np.float)

        for m, a0 in enumerate(an0):
            for n, a1 in enumerate(an1):
                score = self.match_two_annots(a0, a1, n_pairs=len(an0))
                arr[m, n] = score
            a0.parent_image.clear_cache(clear_annot_cached_conv=False)

        edges = []
        while np.sum(arr) > 0:
            i, j = np.unravel_index(arr.argmax(), arr.shape)
            node_0 = '%s|%i' % (im0.filename, i)
            node_1 = '%s|%i' % (im1.filename, j)
            edges.append((node_0, node_1, arr[i, j]))
            arr[i, :] = 0
            arr[:, j] = 0
        return edges, nodes

    def match_two_annots(self, a0: Annotation,
                         a1: Annotation,
                         n_pairs=None,
                         score_threshold=0.50,
                         cache_conv_a0=True,
                         cache_conv_a1=True,
                         cache_conv_a0_im1=True):

        arr1 = a1.parent_image.read(cache=True)

        delta_t = (a1.datetime - a0.datetime).total_seconds()
        if delta_t <= 0 or delta_t > self._max_delta_t:
            return 0.0
        with torch.no_grad():
            d = DataEntry(a0, a1, arr1, n_pairs, net_for_cache=self._net)
            score, conv0, conv1, conv0_1 = self._net(d.as_dict(add_dim=True))
            if cache_conv_a0 and self._net not in a0.cached_conv:
                a0.set_cached_conv(self._net, conv0)
            if cache_conv_a0_im1 and (self._net, a1.datetime) not in a0.cached_conv:
                a0.set_cached_conv((self._net, a1.datetime), conv0_1)

            if cache_conv_a1 and self._net not in a1.cached_conv:
                a1.set_cached_conv(self._net, conv1)

            score = score.item()

        if score < score_threshold:
            score = 0.0
        return score

    @lru_cache(maxsize=None)
    def match_two_tuboids(self, tub0, tub1, score_threshold=0.25):

        # tub0 is the shortest
        if len(tub0) > len(tub1):
            tub0, tub1 = tub1, tub0

        scores = []
        for i, _ in enumerate(tub0):
            min_dt_idx_value_prev = None
            min_dt_idx_value_next = None
            t0 = tub0[i].datetime
            for j, _ in enumerate(tub1[:-1]):
                t1 = tub1[j].datetime
                current = (t1 - t0).total_seconds()

                if current < 0 and (min_dt_idx_value_prev is None or abs(current) < min_dt_idx_value_prev[1]):
                    min_dt_idx_value_prev = j, abs(current)
                if current > 0 and (min_dt_idx_value_next is None or abs(current) < min_dt_idx_value_next[1]):
                    min_dt_idx_value_next = j, abs(current)

            if min_dt_idx_value_next:
                if t0 < t1:
                    tub_a, tub_b = tub0[i], tub1[min_dt_idx_value_next[0]]
                else:
                    tub_b, tub_a = tub0[i], tub1[min_dt_idx_value_next[0]]
                scores.append(self.match_two_annots(tub_a, tub_b))
            if min_dt_idx_value_prev:
                if t0 < t1:
                    tub_a, tub_b = tub0[i], tub1[min_dt_idx_value_prev[0]]
                else:
                    tub_b, tub_a = tub0[i], tub1[min_dt_idx_value_prev[0]]
                scores.append(self.match_two_annots(tub_a, tub_b))

        assert len(scores) > 0
        score = np.mean(scores)
        if score < score_threshold:
            score = 0.0
        return score

    def match_torch_batch(self, data: Dict[str, torch.Tensor]):
        return self._net(data)[0]

