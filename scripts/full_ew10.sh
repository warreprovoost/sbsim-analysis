#!/bin/bash -l
#PBS -N rl_full_ew
#PBS -l walltime=6:30:00
#PBS -l mem=12gb
#PBS -l nodes=1:ppn=4
#PBS -o /user/gent/453/vsc45342/thesis/logs/full_ew.o$PBS_JOBID
#PBS -e /user/gent/453/vsc45342/thesis/logs/full_ew.e$PBS_JOBID
#PBS -m abe

set -e

ml purge
ml GCCcore/12.3.0
ml TensorFlow/2.15.1-foss-2023a
ml load PyTorch-bundle/2.1.2-foss-2023a
ml load Stable-Baselines3/2.3.2-foss-2023a
ml load FFmpeg/6.0-GCCcore-12.3.0
ml load glew/2.2.0-GCCcore-12.3.0-osmesa
ml load protobuf-python/4.24.0-GCCcore-12.3.0

eval "$(conda shell.bash hook)"
conda activate sbsim

mkdir -p /user/gent/453/vsc45342/thesis/results
mkdir -p /user/gent/453/vsc45342/thesis/logs

cd ~/thesis/sbsim-analysis/
SEED=${SEED:-42}
WEIGHT=${WEIGHT:-2.0}
ACTION_DESIGN=${ACTION_DESIGN:-reheat_per_zone}
~/.conda/envs/sbsim/bin/python scripts/train_rl.py \
    --mode full --algo sac --seed ${SEED} \
    --floorplan office_4room \
    --energy_weight ${WEIGHT} \
    --action_design ${ACTION_DESIGN} \
    --unique_run \
    --weather_csv /user/gent/453/vsc45342/thesis/weather_data/oslo_weather_multiyear.csv

echo "Job completed successfully"
