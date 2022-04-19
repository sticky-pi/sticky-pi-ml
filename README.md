# The Machine Learning for the [Sticky Pi project](https://sticky-pi.github.io)

--------------------------------
## Project organisation:

* `src` -- the Python package (`sticky_pi_ml`) that defines all the tools, data structure and implement all algorithms as described in the publication.
* `jobs` -- a set of jobs to train and apply ML models on our dataset. These are slurm jobs that wrap around the package's tools to be run on HPC platforms.


## Installation

### Requierements:

1. `python 3`
2. the `torch` and `torchvision` packages. Their versions **must** be [compatible](https://github.com/pytorch/vision/)
3. the `detectron2` package. [See installation instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

Example to install the above dependencies on a [python virtual environment](https://docs.python.org/3/library/venv.html), in a linux system, using torch 1.10:

```sh
pip install torch==1.10.0 torchvision==0.11.1
python -m pip install detectron2 -f  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html
```

### Installing `sticky-py-ml`

```sh
python -m pip install 'git+https://github.com/sticky-pi/sticky-pi-ml@main#egg=sticky_pi_ml&subdirectory=src'
```

More information on [the central documentation](https://doc.sticky-pi.com/ml.html).