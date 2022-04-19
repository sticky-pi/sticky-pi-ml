import logging
import os

from sticky_pi_api.client import RemoteClient
from sticky_pi_ml.siamese_insect_matcher.ml_bundle import ClientMLBundle
from sticky_pi_ml.siamese_insect_matcher.matcher import Matcher
from sticky_pi_ml.image import ImageSeries

BUNDLE_NAME = 'siamese-insect-matcher'
CANDIDATE_DIR = "candidates"
PREDICT_VIDEO_DIR = "videos"
VALIDATION_OUT_DIR = 'validation_results'


def make_series(i: int):
    import pandas as pd
    csv_file = '../jobs/series.csv'
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


# we set the logging level to "INFO" and show time and file line. nice to prototype
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

client = RemoteClient( "/tmp/test2/",
                       os.environ["API_HOST"],
                       os.environ["API_USER"],
                       os.environ["API_PASSWORD"])


ml_bundle = ClientMLBundle("/home/quentin/Desktop/ml_bundles/siamese-insect-matcher", client, device="cpu")
matcher = Matcher(ml_bundle)

for s in make_series(0):
    print(s)
    out = matcher.match_client(s, video_dir = PREDICT_VIDEO_DIR)
