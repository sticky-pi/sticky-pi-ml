import logging
import os
from sticky_pi_ml.utils import MLScriptParser
from sticky_pi_api.client import LocalClient, RemoteClient
from sticky_pi_ml.siamese_insect_matcher.ml_bundle import ClientMLBundle
from sticky_pi_ml.siamese_insect_matcher.trainer import Trainer
from sticky_pi_ml.siamese_insect_matcher.predictor import Predictor
from sticky_pi_ml.siamese_insect_matcher.matcher import Matcher
from sticky_pi_ml.siamese_insect_matcher.candidates import make_candidates
from sticky_pi_ml.image import ImageSeries

BUNDLE_NAME = 'siamese-insect-matcher'
CANDIDATE_DIR = "candidates"
PREDICT_VIDEO_DIR = "videos"
VALIDATION_OUT_DIR = 'validation_results'


def make_client(opt_dict):
    if opt_dict['local_api']:
        out = LocalClient(opt_dict['LOCAL_CLIENT_DIR'])
    else:
        out = RemoteClient(opt_dict['LOCAL_CLIENT_DIR'],
                           opt_dict['API_HOST'],
                           opt_dict['API_USER'],
                           opt_dict['API_PASSWORD'])
    return out


def make_series(i: int):
    import pandas as pd
    csv_file = '../series.csv'
    df = pd.read_csv(csv_file)
    assert 'device' in df.columns
    assert 'start_datetime' in df.columns
    assert 'end_datetime' in df.columns
    df = df[['device', 'start_datetime', 'end_datetime']]
    if i is None:
        return [ImageSeries(**r) for r in df.to_dict('records')]
    else:
        assert i < len(df)
        return [ImageSeries(**df.iloc[i].to_dict())]

if __name__ == '__main__':
    parser = MLScriptParser()
    option_dict = parser.get_opt_dict()
    bundle_dir = os.path.join(option_dict['BUNDLE_ROOT_DIR'], BUNDLE_NAME)
    ml_bundle_cache = os.path.join(bundle_dir,'.cache')
    os.makedirs(ml_bundle_cache, exist_ok=True)


    if option_dict['action'] == 'fetch':
        client = make_client(option_dict)
        ml_bundle = ClientMLBundle(bundle_dir, client, device=option_dict['device'])
        ml_bundle.sync_remote_to_local()

    elif option_dict['action'] == 'candidates':
        client = make_client(option_dict)
        os.makedirs(CANDIDATE_DIR, exist_ok=True)
        make_candidates(client, out_dir=CANDIDATE_DIR)


    elif option_dict['action'] == 'qc':
        raise NotImplementedError

    elif option_dict['action'] == 'validate':
        client = make_client(option_dict)
        # client = LocalClient(option_dict['LOCAL_CLIENT_DIR'])
        ml_bundle = ClientMLBundle(bundle_dir, client, device=option_dict['device'], cache_dir=ml_bundle_cache)
        t = Trainer(ml_bundle)
        predictor = Predictor(ml_bundle)
        t.validate(predictor, VALIDATION_OUT_DIR)


    elif option_dict['action'] == 'train':
        client = make_client(option_dict)
        ml_bundle = ClientMLBundle(bundle_dir, client, device=option_dict['device'], cache_dir=ml_bundle_cache)
        t = Trainer(ml_bundle)
        t.resume_or_load(resume=not option_dict['restart_training'])
        t.train()

    elif option_dict['action'] == 'predict':
        # this is to analyse series in a slurm job aray
        os.makedirs(PREDICT_VIDEO_DIR, exist_ok=True)
        try:
            i = int(os.getenv('SLURM_ARRAY_TASK_ID'))
            series = make_series(i)
        except (TypeError, ValueError) as e:
            logging.warning('No environment variable named SLURM_ARRAY_TASK_ID. Making series in order instead!')
            series = make_series(None)

        # client = LocalClient(option_dict['LOCAL_CLIENT_DIR'])
        client = make_client(option_dict)
        ml_bundle = ClientMLBundle(bundle_dir, client, device=option_dict['device'], cache_dir=ml_bundle_cache)
        matcher = Matcher(ml_bundle)
        for s in series:
            out = matcher.match_client(s, video_dir = PREDICT_VIDEO_DIR)

    elif option_dict['action'] == 'push':

        client = make_client(option_dict)
        # client = LocalClient(option_dict['LOCAL_CLIENT_DIR'])
        ml_bundle = ClientMLBundle(bundle_dir, client, device=option_dict['device'])
        ml_bundle.sync_local_to_remote()
