# Slurm jobs (e.g. on Compute Canada)

## Install package on a virtual environment
More info on [CC's official doc](https://docs.computecanada.ca/wiki/Python#Creating_and_using_a_virtual_environment)

```shell script
# ADD YOUR OWN VALUE HERE
VIRT_ENV_LOCATION=#~/projects/<FIXME>/sticky_pi_virt_env/

module load python/3.7
module load scipy-stack
module load cuda cudnn

virtualenv --no-download ${VIRT_ENV_LOCATION}
source ${VIRT_ENV_LOCATION}/bin/activate
pip install --no-index --upgrade pip


# we install dependencies by hand :(
# sticky pi client

pip install python-dotenv passlib numpy pandas requests==2.23.0 urllib3-==1.25.9  boto3 pillow psutil joblib sqlalchemy typeguard itsdangerous --no-index
# fail to compile 0.7.4
pip install imread==0.7.0 matplotlib mock 
# this is from branch develop `@develop`
pip install "git+https://github.com/sticky-pi/sticky-pi-api@develop#egg=sticky_pi_api&subdirectory=src" --no-index  --no-deps


# Note that detectron2 is already built as a wheel in ComputeCanada
pip install torch==1.5 pillow-simd detectron2  torchvision==0.6 pycocotools matplotlib mock cython idna chardet decorator cffi tinycss2  markdown   
pip install opencv  svgpathtools CairoSVG networkx shapely==1.6.4.post2 --no-index
pip install "git+https://github.com/sticky-pi/sticky-pi-ml@develop#egg=sticky_pi_ml&subdirectory=src" --no-index  --no-deps
pip install ipython
```

## Define a set of secret environment variables (*not version controlled*).
To be stored in a file named `.secret.env`, in **this directory**. 
Here is a template:

```shell script
export SLURM_ACCOUNT= #<FIXME>

# same value as above
export VIRT_ENV_LOCATION=#~/projects/<FIXME>/sticky_pi_virt_env

# a semi temporary directory or persistent where the training model will end up 
export BUNDLE_ROOT_DIR=#${SCRATCH}/ml_bundles

# persistent directory for the local client.
# will keep all files, images, models, configs...
export LOCAL_CLIENT_DIR=#~/projects/<FIXME>/sticky_pi_client
```
