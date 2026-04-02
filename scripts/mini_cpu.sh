#!/bin/bash -l
#PBS -N rl_mini
#PBS -l walltime=0:30:00
#PBS -l mem=8gb
#PBS -l nodes=1:ppn=4
#PBS -o /user/gent/453/vsc45342/thesis/logs/mini.o$PBS_JOBID
#PBS -e /user/gent/453/vsc45342/thesis/logs/mini.e$PBS_JOBID
#PBS -m abe

set -e  # Exit if any command fails

ml purge
ml GCCcore/12.3.0
ml Python-bundle-PyPI/2023.06-GCCcore-12.3.0
ml TensorFlow/2.15.1-foss-2023a
ml load PyTorch-bundle/2.1.2-foss-2023a
ml load Stable-Baselines3/2.3.2-foss-2023a
ml Gymnasium/0.29.1-foss-2023a

eval "$(conda shell.bash hook)"
conda activate sbsim

mkdir -p /user/gent/453/vsc45342/thesis/results
mkdir -p /user/gent/453/vsc45342/thesis/logs

cd ~/thesis/sbsim-analysis/
~/.conda/envs/sbsim/bin/python scripts/train_rl.py --mode mini --algo sac --seed 42 --unique_run --floorplan office_4room --weather_csv /user/gent/453/vsc45342/thesis/weather_data/oslo_weather_multiyear.csv

echo "Job completed successfully"
