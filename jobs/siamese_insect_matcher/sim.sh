#!/bin/bash
source ../.secret.env

PROGRAM=siamese_insect_matcher.py

echo Training using option "$1"
command="module load cuda cudnn && source ${VIRT_ENV_LOCATION}/bin/activate && ${PROGRAM} $1 -v"

output=$0_$1_$(date +%F_%H-%M-%S)_%N_%j.out
output_array=$0_$1_$(date +%F_%H-%M-%S)_%N_%A_%a.out
JOBNAME=$0.$1

echo Running "$command"

case "$1" in
        candidates)
            sbatch --job-name="${JOBNAME}" --account="${SLURM_ACCOUNT}"  --cpus-per-task=2 --mem=8000M --time=00-02:00 --output="$output" --wrap="${command}"
           ;;
        fetch| push)
            sbatch --job-name="${JOBNAME}" --account="${SLURM_ACCOUNT}" --cpus-per-task=1 --mem=4000M --time=0-05:00 --output="$output" --wrap="${command}"
            ;;
       train)

             sbatch --job-name="${JOBNAME}" --account="${SLURM_ACCOUNT}" --gres=gpu:v100l:1 --cpus-per-task=8 --mem=40000M --time=1-00:00 --output="$output" --wrap="${command} --gpu"
            ;;
       predict)
            sbatch --job-name="${JOBNAME}" --account="${SLURM_ACCOUNT}" --array=0-109%16 --cpus-per-task=4 --mem=16000M --time=0-09:00 --output="$output_array" --wrap="${command}"
#            sbatch --job-name="${JOBNAME}" --account="${SLURM_ACCOUNT}" --cpus-per-task=4 --mem=16000M --time=0-09:00 --output="$output" --wrap="${command}"
           ;;
      validate)
            sbatch --job-name="${JOBNAME}" --account="${SLURM_ACCOUNT}" --gres=gpu:p100:1 --cpus-per-task=2 --mem=16000M --time=00-02:00 --output="$output" --wrap="${command} --gpu"
           ;;
        *)
            echo $"Wrong action"
            exit 1

esac

