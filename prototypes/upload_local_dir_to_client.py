import os
import logging
import glob
from sticky_pi_api.client import LocalClient

DIR_TO_SYNC = "/home/qgeissma/projects/def-juli/qgeissma/legacy/sticky_pi_root/raw_images"
LOCAL_CLIENT_DIR = "/home/qgeissma/projects/def-juli/qgeissma/sticky_pi_client"

DIR_TO_SYNC = "/home/quentin/Desktop/test_raw_images/"
LOCAL_CLIENT_DIR = "/home/quentin/sticky_pi_client"

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

if __name__ == '__main__':
    client = LocalClient(LOCAL_CLIENT_DIR)

    all_images = [f for f in sorted(glob.glob(os.path.join(DIR_TO_SYNC, '**', "*.jpg")))]

    client.put_images(all_images)

    info = [{'device': '%',
             'start_datetime': "1970-01-01_00-00-00",
             'end_datetime': "2070-01-01_00-00-00"}]
    client_resp = client.get_images_with_uid_annotations_series(info, what_image='metadata')



