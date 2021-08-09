#!/bin/bash
source ../.secret.env

PROGRAM=universal_insect_detector.py

echo Training using option "$1"
command="module load cuda cudnn && source ${VIRT_ENV_LOCATION}/bin/activate && ${PROGRAM} $1 -v"

output=$0_$1_$(date +%F_%H-%M-%S)_%N_%j.out
JOBNAME=$0.$1

echo Running "$command"

case "$1" in
        fetch| push)
            sbatch --job-name="${JOBNAME}" --account="${SLURM_ACCOUNT}" --cpus-per-task=1 --mem=4000M --time=0-05:00 --output="$output" --wrap="${command}"
            ;;
       train)
             sbatch --job-name="${JOBNAME}"   --account="${SLURM_ACCOUNT}" --gres=gpu:v100l:1 --cpus-per-task=8 --mem=40000M --time=1-00:00 --output="$output" --wrap="${command} --gpu"
            ;;
       predict)
            sbatch --job-name="${JOBNAME}"   --account="${SLURM_ACCOUNT}" --gres=gpu:p100:1 --cpus-per-task=2 --mem=16000M --time=1-00:00 --output="$output" --wrap="${command} --gpu"
           ;;
      validate)
            sbatch --job-name="${JOBNAME}"   --account="${SLURM_ACCOUNT}" --gres=gpu:p100:1 --cpus-per-task=2 --mem=16000M --time=00-02:00 --output="$output" --wrap="${command} --gpu"
           ;;
        *)
            echo $"Wrong action"
            exit 1

esac

