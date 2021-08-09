import os
from sticky_pi_ml.utils import MLScriptParser
from sticky_pi_api.client import LocalClient, RemoteClient
from sticky_pi_ml.insect_tuboid_classifier.ml_bundle import ClientMLBundle
from sticky_pi_ml.insect_tuboid_classifier.trainer import Trainer
from sticky_pi_ml.insect_tuboid_classifier.predictor import Predictor


BUNDLE_NAME = 'insect-tuboid-classifier'
VALIDATION_OUT_DIR = 'validation_results'
PREDICTION_OUT_DIR = 'prediction_results'

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
    ml_bundle_cache = os.path.join(bundle_dir, '.cache')
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
        predictor = Predictor(ml_bundle)
        t.validate(predictor, VALIDATION_OUT_DIR)

    elif option_dict['action'] == 'train':
        client = make_client(option_dict)
        ml_bundle = ClientMLBundle(bundle_dir, client, device=option_dict['device'], cache_dir=ml_bundle_cache)
        t = Trainer(ml_bundle)
        t.resume_or_load(resume=not option_dict['restart_training'])
        t.train()

    elif option_dict['action'] == 'predict':
        client = make_client(option_dict)
        ml_bundle = ClientMLBundle(bundle_dir, client, device=option_dict['device'], cache_dir=ml_bundle_cache)
        predictor = Predictor(ml_bundle)
        predictor.predict_client(device="%", start_datetime="2020-06-01_00-00-00", end_datetime="2100-01-01_00-00-00",
                                 output_dir=PREDICTION_OUT_DIR)


    elif option_dict['action'] == 'push':
        client = make_client(option_dict)
        ml_bundle = ClientMLBundle(bundle_dir, client, device=option_dict['device'])
        ml_bundle.sync_local_to_remote()
