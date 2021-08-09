import requests
import tempfile
import shutil
import os
import numpy as np
import torch
import logging
from typing import Dict, Tuple, List, Any
import pandas as pd
from sticky_pi_api.types import InfoType
from sticky_pi_ml.annotations import Annotation
from sticky_pi_ml.predictor import BasePredictor
from sticky_pi_ml.insect_tuboid_classifier.model import ResNetPlus, make_resnet
from sticky_pi_ml.tuboid import TiledTuboid
from sticky_pi_ml.insect_tuboid_classifier.ml_bundle import MLBundle, ClientMLBundle
from sticky_pi_ml.insect_tuboid_classifier.dataset import OurTorchDataset


class Predictor(BasePredictor):
    # submit to client N at a time predictions
    _client_predict_chunk_size = 64

    def __init__(self, ml_bundle: MLBundle):
        super().__init__(ml_bundle)
        self._net = self._make_net()
        self._taxonomy_mapper = self._ml_bundle.dataset.taxonomy_mapper
        weights = self._ml_bundle.weight_file

        map = None if torch.cuda.is_available() else 'cpu'
        self._net.load_state_dict(torch.load(weights, map_location=map))
        self._net.eval()

    def _make_net(self):
        return make_resnet(pretrained=False, n_classes=self._ml_bundle.dataset.n_classes)

    def predict_client(self, device, start_datetime, end_datetime, display_prediction=False, output_dir=None):
        assert issubclass(type(self._ml_bundle), ClientMLBundle), \
            "This method only works for MLBundles linked to a client"
        client = self._ml_bundle.client
        logging.info('Processing series %s' % [device, start_datetime, end_datetime])

        series = {'device': device,
                  'start_datetime': start_datetime,
                  'end_datetime': end_datetime}

        client_resp = client.get_tiled_tuboid_series_itc_labels([series], what='data')

        if len(client_resp) < 1:
            logging.warning('No tuboids in %s' % series)
            return

        tiled_tuboids_for_series = pd.DataFrame(client_resp)

        # the fields from  the insect tuboid classifier are suffixed by `_itc`. not to be confused with the fields from
        # the tiled tuboids table
        if 'algo_version_itc' not in tiled_tuboids_for_series.columns:
            logging.info('no labels for this series %s. All %i tuboids will be labeled' %
                         (series, len(tiled_tuboids_for_series)))

            tiled_tuboids_for_series['algo_version_itc'] = None
            tiled_tuboids_for_series['algo_name_itc'] = ""


        initial_n_rows = len(tiled_tuboids_for_series)

        tiled_tuboids_for_series = tiled_tuboids_for_series.sort_values(by=['algo_version_itc', 'start_datetime'])
        tiled_tuboids_for_series = tiled_tuboids_for_series.drop_duplicates(subset=['tuboid_id'], keep='last')

        final_n_rows = len(tiled_tuboids_for_series)
        logging.info(f'{final_n_rows} Unique tuboids({initial_n_rows - final_n_rows} duplicated)')
        initial_n_rows = final_n_rows

        # we want to label tuboids that are not labeled yet by this algorithm (name) or this version of the algo
        conditions = (self.version > tiled_tuboids_for_series.algo_version_itc) | \
                     (tiled_tuboids_for_series.algo_version_itc.isnull()) | \
                     (self.name != tiled_tuboids_for_series.algo_name_itc)

        tiled_tuboids_for_series = tiled_tuboids_for_series[conditions]
        final_n_rows = len(tiled_tuboids_for_series)
        logging.info(f'{final_n_rows} tuboids to annotate ({initial_n_rows - final_n_rows} already annotated with the same algorithm/version)')


        if len(tiled_tuboids_for_series) == 0:
            logging.warning('No tuboids to label in %s (all labeled)' % series)
            return

        def _predict_single_client_tuboid(args):

            r = args[1][1]
            temp_dir = tempfile.mkdtemp()
            try:
                tuboid_dir = os.path.join(temp_dir, r['tuboid_id'])
                os.makedirs(tuboid_dir)
                logging.info(f'Classifying tuboid: {args[0]}/{len(tiled_tuboids_for_series)}: {r["tuboid_id"]}')
                for f in ['metadata', 'tuboid', 'context']:
                    if os.path.isfile(r[f]):
                        shutil.copy(r[f], tuboid_dir)
                    else:
                        filename = os.path.basename(r[f]).split('?')[0]
                        resp = requests.get(r[f]).content
                        with open(os.path.join(tuboid_dir, filename), 'wb') as file:
                            file.write(resp)

                tiled_tuboid = TiledTuboid(tuboid_dir)
                prediction = self.predict(tiled_tuboid)
                if display_prediction:
                    self._display_prediction(prediction, tiled_tuboid)
                if output_dir:
                    self._make_prediction_image(prediction, tiled_tuboid,output_dir)

            finally:
                shutil.rmtree(temp_dir)

            prediction['algo_version'] = self.version
            prediction['algo_name'] = self.name
            prediction['tuboid_id'] = r['tuboid_id']
            logging.info('Prediction: %s' % prediction)
            client.put_itc_labels([prediction])

        # from multiprocessing.pool import ThreadPool
        # pool = ThreadPool(1)
        # pool.map(_predict_single_client_tuboid, enumerate(tiled_tuboids_for_series.iterrows()))

        for i in  enumerate(tiled_tuboids_for_series.iterrows()):
            _predict_single_client_tuboid(i)


    def _make_prediction_image(self, prediction: Dict, tiled_tuboid: TiledTuboid, output_dir):
        import cv2
        from threading import current_thread
        im = tiled_tuboid.get_tile(0)['array']
        h, w, _ = im.shape
        im = np.zeros((h + 128, w * 5, 3), np.uint8)
        for t, i in zip(tiled_tuboid.iter_tiles(), range(0, 5)):
            im[0:h, i * w: (i + 1) * w:] = t['array']
        cv2.putText(im, prediction['pattern'], (24, 64 + h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1,
                    cv2.LINE_AA)
        target = os.path.join(output_dir, os.path.basename(tiled_tuboid.directory) + '.jpg', )
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(target, im)

    def _display_prediction(self, prediction: Dict, tiled_tuboid: TiledTuboid):
        import cv2
        from threading import current_thread
        im = tiled_tuboid.get_tile(0)['array']
        h , w, _ = im.shape
        im = np.zeros((h + 128, w * 5, 3), np.uint8)
        for t, i in zip(tiled_tuboid.iter_tiles(), range(0, 5)):
            im[0:h, i * w: (i+1)*w:] = t['array']
        cv2.putText(im, prediction['pattern'], (24, 64+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1,cv2.LINE_AA)
        cv2.imshow(current_thread().name, im)
        cv2.waitKey(1)



    def predict(self, tiled_tuboid: TiledTuboid):
        data_entry = OurTorchDataset.tiled_tuboid_to_dict(tiled_tuboid, unsqueezed=True)
        preds = self._net(data_entry)
        preds = preds.detach().numpy()
        label = int(np.argmax(preds))

        out = self._taxonomy_mapper.label_to_level_dict(label)
        pattern = self._taxonomy_mapper.label_to_pattern(label)
        out.update({'label': label,
                'pattern': pattern,
                })
        return out
