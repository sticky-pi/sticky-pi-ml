#!/bin/bash
# send the tuboids to the s3 for manual annotation 
source ../.secret.env

export BUCKET_NAME=tuboid-annotation-data
export TUBOID_DATA_DIR=~/projects/def-juli/qgeissma/sticky_pi_client/tiled_tuboids

command="module load cuda cudnn && source ${VIRT_ENV_LOCATION}/bin/activate && s3cmd sync ${TUBOID_DATA_DIR}/ s3://${BUCKET_NAME}"

output=$0_$1_$(date +%F_%H-%M-%S)_%N_%j.out
JOBNAME=$0.$1
echo Running "$command"
sbatch --job-name="${JOBNAME}" --account="${SLURM_ACCOUNT}" --cpus-per-task=1 --mem=4000M --time=0-12:00 --output="$output" --wrap="${command}"
