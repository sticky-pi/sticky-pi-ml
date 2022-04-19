import os

from sticky_pi_api.client import  RemoteClient

import logging
import pandas as pd

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

import sys

if __name__ == '__main__':
    try:
        out_dir = sys.argv[1]
    except IndexError:
        raise IndexError('Must provide an argument for the output dir')

    os.makedirs(out_dir, exist_ok=True)


    args = (os.environ['LOCAL_CLIENT_DIR'],
                                os.environ['API_HOST'],
                                os.environ['API_USER'],
                                os.environ['API_PASSWORD'])
    for a in args:
        assert a is not None
    client = out = RemoteClient(*args)

    series = {'device': "%",
              'start_datetime': "2020-06-01_00-00-00",
              'end_datetime': "2100-01-01_00-00-00"}

    client_resp = client.get_tiled_tuboid_series_itc_labels([series], what='metadata')
    tiled_tuboids_for_series = pd.DataFrame(client_resp)

    tiled_tuboids_for_series = tiled_tuboids_for_series.sort_values(by=['algo_version_itc', 'start_datetime'])
    tiled_tuboids_for_series = tiled_tuboids_for_series.drop_duplicates(subset=['tuboid_id'], keep='last')


    tiled_tuboids_for_series.to_csv(os.path.join(out_dir, 'itc_labels.csv'))
