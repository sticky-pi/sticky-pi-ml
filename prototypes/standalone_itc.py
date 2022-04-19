import shutil
import argparse
import logging
import os
import glob
import pandas as pd
from sticky_pi_ml.insect_tuboid_classifier.ml_bundle import MLBundle
from sticky_pi_ml.insect_tuboid_classifier.trainer import Trainer
from sticky_pi_ml.insect_tuboid_classifier.predictor import Predictor
from sticky_pi_ml.tuboid import TiledTuboid


OUTPUT_FILENAME = "results.csv"

valid_actions = {"predict_dir", "train"}
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
        tuboid_dir = option_dict["target"]
        target_file = os.path.join(option_dict["target"], OUTPUT_FILENAME)
        if os.path.exists(target_file):
            if not option_dict["force"]:
                raise FileExistsError(f"Result file already exists: {target_file}. Use --force to overwrite")

        # check we can write this file
        with open(target_file, "w") as f:
            pass

        logging.info(f"Will generate output as {target_file}")
        ml_bundle = MLBundle(option_dict["bundle_dir"])
        predictor = Predictor(ml_bundle)
        tuboid_metadata = sorted(glob.glob(os.path.join(option_dict['target'], "**", "metadata.txt"), recursive=True))
        out = []
        try:
            for met in tuboid_metadata:
                tt_dir = os.path.dirname(met)
                tt = TiledTuboid(tt_dir)
                prediction = predictor.predict(tt)
                prediction["directory"] = tt_dir

                logging.info(prediction)

                out.append(prediction)
        except Exception as e:
            logging.error(f"Failed to analyse tuboid {tt_dir}. Saving what we have so far.")
            raise e
        finally:
            dt = pd.DataFrame(out)
            logging.info(dt)
            logging.info(f"Saving all data as {target_file}")
            dt.to_csv(target_file)


    if option_dict["action"] == "train":
        ml_bundle = MLBundle(option_dict["bundle_dir"])
        t = Trainer(ml_bundle)
        t.resume_or_load(resume=not option_dict['restart_training'])
        t.train()
