import sys
import tempfile
import pandas as pd
import os
import dotenv
import requests
from sticky_pi_api.client import LocalClient, RemoteClient
from sticky_pi_ml.image import ImageJsonAnnotations
from sticky_pi_ml.utils import md5

PROPORTION_IMAGE_TO_KEEP = 5.0 / 100.0
OUTPUT = "uid_candidates"

if __name__ == '__main__':
    dotenv.load_dotenv("../jobs/.secret.env")

    client = RemoteClient(os.environ['LOCAL_CLIENT_DIR'],
                          os.environ['API_HOST'],
                          os.environ['API_USER'],
                          os.environ['API_PASSWORD'])

    o = client.get_images_with_uid_annotations_series([{"device": "%",
                                                        "start_datetime": "2021-06-07_00-00-00",
                                                        "end_datetime": "2021-06-11_00-00-00"}],
                                                      what_image="metadata",
                                                      what_annotation="metadata"
                                                      )
    df = pd.DataFrame(o)

    df = df[df.id_annot.isnull() == False]
    # we take images whose md5 is in the lower 5%

    max_md5 = int('ffff', base=16)
    df["to_fetch"] = df.apply(lambda x: int(x.md5[0:4], base=16) / max_md5 < PROPORTION_IMAGE_TO_KEEP, axis=1)

    df = df[df.to_fetch]
    image_query = df[["datetime", "device"]].to_dict(orient="record")
    im_df = pd.DataFrame(client.get_images(image_query, what='image'))
    an_df = pd.DataFrame(client.get_uid_annotations(image_query, what='data'))
    df = pd.merge(im_df, an_df, left_on="id", right_on="parent_image_id")
    temp = tempfile.mkdtemp()
    os.makedirs(OUTPUT, exist_ok=True)
    try:
        for i, r in df.iterrows():
            r_dict = r.to_dict()
            filename = os.path.basename(r_dict['url']).split('?')[0]
            target = os.path.join(temp, filename)
            print(f"Downloading {filename}")
            if os.path.isfile(target):
                local_md5 = md5(target)
            else:
                local_md5 = None

            if r_dict['md5'] != local_md5:
                resp = requests.get(r_dict['url']).content
                with open(target, 'wb') as file:
                    file.write(resp)

            im = ImageJsonAnnotations(target, json_str=r_dict["json"])
            svg_target = os.path.join(OUTPUT, os.path.splitext(os.path.basename(target))[0] + ".svg")
            print(f"Saving {svg_target}")
            im.to_svg(svg_target)

    finally:
        import shutil

        shutil.rmtree(temp)
