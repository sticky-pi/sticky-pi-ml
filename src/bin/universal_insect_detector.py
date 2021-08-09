import os
from sticky_pi_ml.universal_insect_detector.ml_bundle import ClientMLBundle
from sticky_pi_ml.utils import MLScriptParser
# fixme
from sticky_pi_api.client import LocalClient, RemoteClient
from sticky_pi_ml.universal_insect_detector.trainer import Trainer
from sticky_pi_ml.universal_insect_detector.predictor import Predictor

BUNDLE_NAME = 'universal-insect-detector'
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

    elif option_dict['action'] == 'qc':
        raise NotImplementedError

    elif option_dict['action'] == 'validate':
        client = make_client(option_dict)
        ml_bundle = ClientMLBundle(bundle_dir, client, device=option_dict['device'], cache_dir=ml_bundle_cache)
        t = Trainer(ml_bundle)
        pred = Predictor(ml_bundle)
        os.makedirs(VALIDATION_OUT_DIR, exist_ok=True)
        t.validate(pred, out_dir=VALIDATION_OUT_DIR)


    elif option_dict['action'] == 'train':
        client = make_client(option_dict)
        ml_bundle = ClientMLBundle(bundle_dir, client, device=option_dict['device'], cache_dir=ml_bundle_cache)
        t = Trainer(ml_bundle)
        t.resume_or_load(resume=not option_dict['restart_training'])
        t.train()

    elif option_dict['action'] == 'predict':
        client = make_client(option_dict)
        ml_bundle = ClientMLBundle(bundle_dir, client, device=option_dict['device'], cache_dir=ml_bundle_cache)
        pred = Predictor(ml_bundle)
        pred.detect_client()


    elif option_dict['action'] == 'push':
        client = make_client(option_dict)
        ml_bundle = ClientMLBundle(bundle_dir, client, device=option_dict['device'])
        ml_bundle.sync_local_to_remote()
