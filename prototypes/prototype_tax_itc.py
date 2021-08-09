"""
```sh
# To run in terminal
ML_BUNDLE_DIR=<A LOCAL DIRECTORY FOR THE BUNDLE>

# This could take a while (ensure you have stable network.. ideally overnight or so).
s3cmd sync s3://sticky-pi-api-prod/ml/insect-tuboid-classifier/ ${ML_BUNDLE}/
```
"""
import os
import logging
from sticky_pi_ml.insect_tuboid_classifier.ml_bundle import MLBundle as OriginalMLBundle
from sticky_pi_ml.insect_tuboid_classifier.trainer import Trainer as OriginalTrainer
from sticky_pi_ml.insect_tuboid_classifier.dataset import Dataset as OriginalDataset


# we set the logging level to "INFO" and show time and file line. nice to prototype
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)


ML_BUNDLE_DIR = "/home/quentin/Desktop/insect-tuboid-classifier"#fixme same as above

assert os.path.isdir(ML_BUNDLE_DIR)

class Dataset(OriginalDataset):
    # here, you could reimplement your own methods e.g. to filter certain classes
    pass

class MLBundle(OriginalMLBundle):
    _name = 'taxonomic-itc'
    _DatasetClass = Dataset

class Trainer(OriginalTrainer):
    pass

    # to overide base method:
    # def train(self):
    #   work here ;) good luck


bndl = MLBundle(ML_BUNDLE_DIR)
trainer = Trainer(bndl)

# This next command should print statistics about the data 'PATTERN (LABEL) -> N', where LABEL  is an integer (>=0),
# N, the number of individual in the dataset, PATTERN, a regex matching "the label"
trainer.resume_or_load(resume=True)

trainer.train()


