#!/bin/bash
#SBATCH --job-name=rl_long_new
#SBATCH --time=12:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=/user/gent/453/vsc45342/thesis/logs/long.o%j
#SBATCH --error=/user/gent/453/vsc45342/thesis/logs/long.e%j
#SBATCH --mail-type=BEGIN,END,FAIL

ml purge
ml GCCcore/12.3.0
ml Python-bundle-PyPI/2023.06-GCCcore-12.3.0
ml PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

export PYTHONPATH="/user/gent/453/vsc45342/.conda/envs/sbsim/lib/python3.11/site-packages:/apps/gent/RHEL9/cascadelake-ampere-ib/software/PyTorch/2.1.2-foss-2023a-CUDA-12.1.1/lib/python3.11/site-packages"
export LD_LIBRARY_PATH=/apps/gent/RHEL9/cascadelake-ib/software/GCCcore/12.3.0/lib64:$LD_LIBRARY_PATH

eval "$(conda shell.bash hook)"
conda activate sbsim

mkdir -p /user/gent/453/vsc45342/thesis/results
mkdir -p /user/gent/453/vsc45342/thesis/logs

cd ~/thesis/sbsim-analysis/
SEED=${SEED:-42}
WEIGHT=${WEIGHT:-2.0}
ACTION_DESIGN=${ACTION_DESIGN:-reheat_per_zone}
ALGO=${ALGO:-sac}

IS_RESUBMIT=${IS_RESUBMIT:-0}
# Print params in a parseable format so check_jobs.sh can resubmit if cancelled
echo "JOB_PARAMS: ALGO=${ALGO} WEIGHT=${WEIGHT} SEED=${SEED} ACTION_DESIGN=${ACTION_DESIGN} IS_RESUBMIT=${IS_RESUBMIT}"

# Check if GPU is already heavily used (shared node with another job)
gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
if [ -n "$gpu_util" ] && [ "$gpu_util" -gt 20 ]; then
    echo "GPU_SHARING: GPU already at ${gpu_util}% utilization — check_jobs.sh will cancel and resubmit"
else
    echo "GPU_SHARING: GPU at ${gpu_util:-?}% utilization, proceeding"
fi
~/.conda/envs/sbsim/bin/python scripts/train_rl.py \
    --mode long --algo ${ALGO} --seed ${SEED} \
    --floorplan office_4room \
    --energy_weight ${WEIGHT} \
    --action_design ${ACTION_DESIGN} \
    --unique_run \
    --weather_csv /user/gent/453/vsc45342/thesis/weather_data/belgium_weather_multiyear.csv \

echo "Job completed successfully"
