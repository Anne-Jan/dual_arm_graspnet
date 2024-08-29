#!/bin/bash
#SBATCH --job-name="da2graspnet-train-example"
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=v100:1
#SBATCH --output=job-%j.log

cd /projects/s3399834
module load Anaconda3/2022.05
conda activate DA2graspnet
module load CUDA/11.7.0

echo '======example usage======='
python python3 -m graspnetbackup.train.py  --arch evaluator  --dataset_root_folder shapenet_models/da2_dataset/ --batch_size 192 --num_grasps_per_object 192 --niter 1000 --niter_decay 10000 --save_epoch_freq 10 --save_latest_freq 50 --run_test_freq 10 --dual_grasp True --num_threads 3 --continue_train