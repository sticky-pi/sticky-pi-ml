import os
import pandas as pd
import tempfile
import shutil
import logging
from functools import lru_cache
import numpy as np
import networkx as nx
from typing import List

from sticky_pi_ml.siamese_insect_matcher.predictor import Predictor
from sticky_pi_ml.siamese_insect_matcher.ml_bundle import MLBundle, ClientMLBundle
from sticky_pi_ml.tuboid import Tuboid, TiledTuboid
from sticky_pi_ml.image import ImageSeries


class Matcher(object):
    _tub_min_length = 3

    def __init__(self, ml_bundle: MLBundle, PredictorClass=Predictor):
        self._predictor = PredictorClass(ml_bundle)
        self._ml_bundle = ml_bundle

    def match_client(self, annotated_images_series: ImageSeries, video_dir: str = None):
        series_info = annotated_images_series.info_dict
        series_info.update({'algo_name': self._ml_bundle.name,
                            'algo_version': self._ml_bundle.version})

        assert issubclass(type(self._ml_bundle), ClientMLBundle), \
            "This method only works for MLBundles linked to a client"
        client = self._ml_bundle.client
        logging.info('Processing series %s' % annotated_images_series)

        temp_dir = tempfile.mkdtemp(suffix=annotated_images_series.name)
        try:
            cache_image_dir = os.path.join(temp_dir, "cache_images")
            os.makedirs(cache_image_dir)
            already_present_tuboids = pd.DataFrame(client.get_tiled_tuboid_series([annotated_images_series.info_dict]))
            if 'algo_version' in already_present_tuboids.columns:
                same_algo_tuboids = already_present_tuboids[
                    already_present_tuboids.algo_version_series == self._ml_bundle.version]
                n_same_algo = len(same_algo_tuboids)
                if n_same_algo > 0:
                    expected_n_same_algo = same_algo_tuboids.iloc[0].n_tuboids_series
                    if expected_n_same_algo != n_same_algo:
                        logging.error(f"Upstream tuboid series partially uploaded: expected {expected_n_same_algo}, but only contains {n_same_algo}")
                        return
                    else:
                        logging.info('Series %s already been analysed by the same algorithm, and is complete.'
                                     'Skipping.' % annotated_images_series)
                        return
            annotated_images_series.populate_from_client(client, cache_image_dir=cache_image_dir)

            if len(annotated_images_series) < 3:
                logging.warning('Only %i annotated images in %s. Need 3 at least!' % (
                    len(annotated_images_series), annotated_images_series))
            tuboids = self.match(annotated_images_series)

            tuboid_temp_dir = os.path.join(temp_dir, "cache_tuboids")
            os.makedirs(tuboid_temp_dir)
            to_upload = [TiledTuboid.from_tuboid(t, tuboid_temp_dir).directory for t in tuboids]

            series_info['n_images'] = len(annotated_images_series)
            series_info['n_tuboids'] = len(to_upload)

            client.put_tiled_tuboids(to_upload, series_info=series_info)
            logging.info('Series %s uploaded!' % annotated_images_series)
            if video_dir is not None:
                self.make_video(tuboids, os.path.join(video_dir, annotated_images_series.name + '.mp4'),
                                annotated_images_series=annotated_images_series)

            logging.info('Done with series %s' % annotated_images_series)
            return tuboids
        finally:
            shutil.rmtree(temp_dir)

    def match(self, annotated_images_series: ImageSeries) -> List[Tuboid]:
        assert len(annotated_images_series) > 2
        try:
            tuboids = self._draft_graph(annotated_images_series)
            logging.info('%i Stitching non-contiguous tuboids head to tail' % len(tuboids))
            tuboids = self._stitch_sub_graphs(tuboids, annotated_images_series)

            logging.info('%i tuboid detected. Merging conjoint tuboids' % len(tuboids))
            tuboids = self._merge_conjoint_tuboids(tuboids, annotated_images_series)

            logging.info('%i tuboid left. Removing small tuboids' % len(tuboids))
            tuboids = [tub for tub in tuboids if len(tub) > self._tub_min_length]

            logging.info('%i tuboid left. Cleaning up' % len(tuboids))
            # reordering tuboids
            Tuboid.set_instances_id(tuboids)
        finally:
            self._clean_up(annotated_images_series)
        return tuboids

    def _draft_graph(self, annotated_images: ImageSeries) -> List[Tuboid]:
        dg = nx.DiGraph()
        annotated_images.sort(key=lambda x: x.datetime)
        s0, s1 = annotated_images[0:2]
        for i, _ in enumerate(annotated_images[1:-1]):
            logging.info('Drafting, %i/%i; %i edges in graph' % (i + 1, len(annotated_images) - 2, len(dg.edges)))
            s1 = annotated_images[i + 1]
            edges, nodes = self._predictor.match_two_images(s0, s1)

            # add the resulting nodes and edges to the overall graph
            # we define these primary edges as not stitched (as opposed to edges that span multiple frames)
            dg.add_weighted_edges_from(edges, stitched=False)
            # only add nodes that do not exist
            for k, v in nodes.items():
                if k not in dg.nodes:
                    dg.add_node(k)
                dg.nodes[k].update({'annotation': v})
            # optimisation: the first image had been cached so that
            # matching / extraction of the sub-image can be faster. For now, we clear the cache
            s0.clear_cache()
            s0 = s1

        s1.clear_cache()

        tuboids = []
        for sub_graph in nx.weakly_connected_components(dg):
            annotations = [nd[1]['annotation'] for nd in dg.subgraph(sub_graph).nodes(data=True)]
            tuboids.append(Tuboid(annotations, self._ml_bundle.version, parent_series=annotated_images))
        return tuboids

    def _stitch_sub_graphs(self, tuboids: List[Tuboid], annotated_images_series: ImageSeries) -> List[Tuboid]:
        ntub = len(tuboids)
        if ntub < 2:
            return tuboids
        arr = np.zeros((ntub, ntub), dtype=np.float)
        for i in range(ntub):
            for j in range(i + 1, ntub):
                score = self._predictor.match_two_annots(tuboids[i].tail,
                                                         tuboids[j].head)
                arr[i, j] = score
            # fixme. release parent image for annot i here? that would free memory?

        edges = []
        while np.sum(arr) > 0.0:
            i, j = np.unravel_index(arr.argmax(), arr.shape)
            edges.append((i, j, arr[i, j]))
            arr[i, :] = 0
            arr[:, j] = 0

        dg = nx.DiGraph()
        dg.add_weighted_edges_from(edges)

        for sdg in nx.weakly_connected_components(dg):
            annots_to_merge = []
            for tb in sdg:
                annots_to_merge.extend(tuboids[tb])
                # place holder to later delete without changing the indices yet
                tuboids[tb] = None
            merged_tuboid = Tuboid(annots_to_merge, self._ml_bundle.version,
                                   parent_series=annotated_images_series)

            tuboids.append(merged_tuboid)
        tuboids = [tb for tb in tuboids if tb is not None]
        return tuboids

    # to tuboids are `conjoint' iff:
    # * their range overlap in time
    # * they have no coincident reads
    # Typically, conjoint tuboids are the same instance, but falsely clustered as multiple ones.
    # the goal is to iteratively merge conjoint tuboids
    def _merge_conjoint_tuboids(self, tuboids, annotated_images_series: ImageSeries, iteration=0):
        ntub = len(tuboids)
        if ntub < 2:
            return tuboids
        logging.info('Merging conjoint tuboids. Iteration %i' % iteration)
        arr = np.zeros((ntub, ntub), dtype=np.float)
        for i in range(ntub):
            tb0 = tuboids[i]
            for j in range(i + 1, ntub):
                tb1 = tuboids[j]
                if self._is_tuboid_pair_conjoint(tb0, tb1):
                    match = self._predictor.match_two_tuboids(tb0, tb1)
                    if match:
                        arr[i, j] = match
        logging.info('n tuboid matches: %i' % np.sum(arr > 0))
        if np.sum(arr) == 0:
            return tuboids

        i, j = np.unravel_index(arr.argmax(), arr.shape)

        assert j > i
        tb1 = tuboids.pop(j)
        tb0 = tuboids.pop(i)

        merged_tuboid = Tuboid(tb1 + tb0, self._ml_bundle.version, parent_series=annotated_images_series)
        tuboids.append(merged_tuboid)

        # we recurse as still need to merge some tuboids
        tuboids = self._merge_conjoint_tuboids(tuboids, annotated_images_series, iteration=iteration + 1)
        return tuboids

    @lru_cache(maxsize=None)
    def _is_tuboid_pair_conjoint(self, tb0, tb1):
        if len(tb0) + len(tb1) < 3:
            return False

        if not (tb0.tail_datetime > tb1.head_datetime > tb0.head_datetime or
                tb1.tail_datetime > tb0.head_datetime > tb1.head_datetime):
            return False

        # todo optimise. lists are presorted!
        for t0 in tb0:
            for t1 in tb1:
                if t1.datetime == t0.datetime:
                    return False
        return True

    def _clean_up(self, annotated_images_series: ImageSeries):
        logging.info('deleting img cache!')
        for im in annotated_images_series:
            im.clear_cache()

    @staticmethod
    def make_video(tuboids, out, annotated_images_series: ImageSeries, scale=(1600, 1200), fps=4, show=False):
        import cv2

        def col_lut(id):
            np.random.seed(id)
            gr = np.round(np.random.random(2) * 255).astype(int).tolist()
            return 255, gr[0], gr[1]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(out, fourcc, fps, (scale[0], scale[1]))
        logging.info('Saving video in %s' % out)

        try:
            for i, s in enumerate(annotated_images_series):
                im = np.copy(s.read())
                logging.info('Processing %s (%i/%i)' % (s.filename, i, len(annotated_images_series)))
                for j, t in enumerate(tuboids):
                    assert s.device == t.device
                    bbox, inferred = t.bbox_at_datetime(s.datetime)
                    if bbox is not None:
                        col = col_lut(j)
                        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color=col,
                                      thickness=3)
                        text = "%03d" % t.id
                        cv2.putText(im, text, (bbox[0] + bbox[2], bbox[1] + bbox[3]), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                    color=col, thickness=3)

                cv2.putText(im, s.filename, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), thickness=3)

                im = cv2.resize(im, scale)
                if show:
                    cv2.imshow('im', im)
                    cv2.waitKey(250)
                vw.write(im)
        finally:
            if vw is not None:
                vw.release()
