#!/bin/bash
#SBATCH --job-name="da2graspnet-train-example"
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=v100:1
#SBATCH --output=job-%j.log

cd /home1/p300488/graspnetbackup
conda activate DA2graspnet
module load CUDA/11.7.0

echo '======example usage======='
python -m demo.main_headless
