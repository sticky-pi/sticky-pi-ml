import shutil
import argparse
import logging
import os
import glob

from sticky_pi_ml.insect_tuboid_classifier.ml_bundle import MLBundle
from sticky_pi_ml.insect_tuboid_classifier.trainer import Trainer
from sticky_pi_ml.insect_tuboid_classifier.predictor import Predictor



valid_actions = {"predict_dir", "train", 'validate'}
if __name__ == '__main__':
    args_parse = argparse.ArgumentParser()
    args_parse.add_argument("action", help=str(valid_actions))
    args_parse.add_argument("-b", "--bundle-dir", dest="bundle_dir")

    args_parse.add_argument("-v", "--verbose", dest="verbose", default=False,
                            help="verbose",
                            action="store_true")

    args_parse.add_argument("-D", "--debug", dest="debug", default=False,
                            help="debug",
                            action="store_true")
    args_parse.add_argument("-g", "--gpu", dest="gpu", default=False, help="Wehther to use GPU/Cuda", action="store_true")

    # predict specific
    args_parse.add_argument("-t", "--target", dest="target")

    args_parse.add_argument("-f", "--force", dest="force", default=False, help="force", action="store_true")
    args_parse.add_argument("-k", "--filter", default=1, help="force", type=int)

    # training specific
    args_parse.add_argument("-r", "--restart-training", dest="restart_training", default=False, action="store_true")


    args = args_parse.parse_args()
    option_dict = vars(args)

    if option_dict['verbose']:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.INFO)
    if option_dict["action"] != "candidates":
        if not option_dict["bundle_dir"]:
            raise ValueError("--bundle-dir (-b) not defined")
        if not os.path.isdir(option_dict["bundle_dir"]):
            raise ValueError(f"--bundle-dir refers to a directory that does NOT exist: {option_dict['bundle_dir']}")

    if  option_dict["action"] == "predict_dir":
        tuboid_dir = os.path.join(option_dict["target"], TUBOID_DIR_NAME)
        if os.path.exists(tuboid_dir):
            if not option_dict["force"]:
                raise FileExistsError(f"result directory already exists: {tuboid_dir}. Use --force to overwrite")
            else:
                shutil.rmtree(tuboid_dir)
        logging.info(f"Will generate tuboids in {tuboid_dir}")
        os.makedirs(tuboid_dir)

        im_series = ImageSeriesSVGDir(option_dict["target"])

        ml_bundle = MLBundle(option_dict["bundle_dir"])
        matcher = Matcher(ml_bundle)

        tuboids = matcher.match(im_series)
        series_id = tuboids[0].parent_series.name + "." + tuboids[0].matcher_version
        tiled_tuboids = [TiledTuboid.from_tuboid(t, tuboid_dir).directory for t in tuboids]

        logging.info("Making video")
        matcher.make_video(tuboids, os.path.join(tuboid_dir,series_id, im_series.name + '.mp4'),
                        annotated_images_series = im_series)

    if option_dict["action"] == "train":
        ml_bundle = MLBundle(option_dict["bundle_dir"])
        t = Trainer(ml_bundle)
        t.resume_or_load(resume=not option_dict['restart_training'])
        t.train()

    elif option_dict['action'] == 'validate':

        ml_bundle = MLBundle(option_dict["bundle_dir"])
        t = Trainer(ml_bundle)
        predictor = Predictor(ml_bundle)
        os.makedirs(option_dict["target"], exist_ok=True)
        t.validate(predictor, option_dict["target"])

    if option_dict["action"] == "candidates":

        assert os.path.isdir(option_dict["target"]), FileNotFoundError(f"Target directory: {option_dict['target']} does not exist")
        files =[f for f in sorted(glob.glob(os.path.join(option_dict["target"], "*.svg")))]
        assert len(files) >1, ValueError(f"Found {len(files)} svg files. At least 2 are needed")

        candidate_dir = os.path.join(option_dict["target"], CANDIDATES_DIR_NAME)
        if os.path.exists(candidate_dir):
            if not option_dict["force"] and os.listdir(candidate_dir):
                raise FileExistsError(f"result directory already exists: {candidate_dir}. Use --force to overwrite")
            else:
                shutil.rmtree(candidate_dir)
        logging.info(f"Will generate candidates in {candidate_dir}")
        os.makedirs(candidate_dir)

        for i, (f0, f1) in enumerate(zip(files[:-1], files[1:])):
            if i % option_dict["filter"] != 0:
                continue
            logging.info((f0, f1))
            im0 = SVGImage(f0)
            im1 = SVGImage(f1)

            logging.info("Candidates Merging %s + %s in %s" % (im0.filename, im1.filename, candidate_dir))
            SiamSVG.merge_two_images(im0, im1, dest_dir=candidate_dir)
