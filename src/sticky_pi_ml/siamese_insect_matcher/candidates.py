import os
import pandas as pd
import random
import logging
from sticky_pi_ml.image import ImageJsonAnnotations
from sticky_pi_ml.siamese_insect_matcher.siam_svg import SiamSVG
from sticky_pi_api.client import LocalClient


def make_candidates(client: LocalClient, out_dir, info=None, every=50, max_delta_t: int = 12 * 3600):
    random.seed(1234)

    assert os.path.isdir(out_dir)

    if info is None:
        info = [{'device': '%',
                 'start_datetime': "1970-01-01_00-00-00",
                 'end_datetime': "2070-01-01_00-00-00"}]
        logging.info('No info provided. Fetching all annotations')

    resp = client.get_images_with_uid_annotations_series(info, what_image='image',
                                                         what_annotation='data')
    df = pd.DataFrame(resp)

    df.sort_values(['datetime'], inplace=True)

    for device, sub_df in df.groupby('device'):
        for i in range(len(sub_df)):
            ri = sub_df.iloc[i]
            target_j = random.randint(1, max_delta_t)
            if i % every == 0:
                logging.info("Match IM0: %s, %s" % (ri.device, ri.datetime))
                logging.info("Target: %s" % str(target_j))

                rj = None
                for j in range(i, len(sub_df)):
                    rj = sub_df.iloc[j]
                    if (rj.datetime-ri.datetime).total_seconds() > target_j:
                        break
                if rj is None:
                    continue
                logging.info("Match IM1: %s, %s" % (rj.device, rj.datetime))

                if not ri.json or not rj.json:
                    logging.warning('No annotations')
                    continue

                im0 = ImageJsonAnnotations(ri.url, ri.json)
                im1 = ImageJsonAnnotations(rj.url, rj.json)
                logging.info("Candidates Merging %s + %s in %s" % (im0.filename, im1.filename, out_dir))
                SiamSVG.merge_two_images(im0, im1, dest_dir=out_dir)
