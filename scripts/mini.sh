#!/bin/bash -l
#PBS -N rl_mini
#PBS -l walltime=0:30:00
#PBS -l mem=8gb
#PBS -l nodes=1:ppn=2:gpus=1
#PBS -o /user/gent/453/vsc45342/thesis/logs/mini.o$PBS_JOBID
#PBS -e /user/gent/453/vsc45342/thesis/logs/mini.e$PBS_JOBID
#PBS -m abe

set -e  # Exit if any command fails

ml purge
ml GCCcore/12.3.0
ml TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
ml load FFmpeg/6.0-GCCcore-12.3.0
ml load glew/2.2.0-GCCcore-12.3.0-osmesa
ml load protobuf-python/4.24.0-GCCcore-12.3.0

eval "$(conda shell.bash hook)"
conda activate sbsim

mkdir -p /user/gent/453/vsc45342/thesis/results
mkdir -p /user/gent/453/vsc45342/thesis/logs

cd ~/thesis/sbsim-analysis/
~/.conda/envs/sbsim/bin/python scripts/train_rl.py --mode mini --algo sac --seed 42 --unique_run

echo "Job completed successfully"
