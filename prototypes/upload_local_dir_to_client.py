import os
import logging
import glob
from sticky_pi_api.client import LocalClient

DIR_TO_SYNC = "/home/qgeissma/projects/def-juli/qgeissma/legacy/sticky_pi_root/raw_images"
LOCAL_CLIENT_DIR = "/home/qgeissma/projects/def-juli/qgeissma/sticky_pi_client"

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

if __name__ == '__main__':
    client = LocalClient(LOCAL_CLIENT_DIR)
    all_images = [f for f in sorted(glob.glob(os.path.join(DIR_TO_SYNC, '**', "*.jpg")))]
    client.put_images(all_images)

