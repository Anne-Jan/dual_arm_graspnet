#!/bin/bash
#SBATCH --job-name="setup-example"
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=v100:1
#SBATCH --output=job-%j.log

cd /home3/s3399834/graspnetbackup
module load CUDA/11.7.0
module load Anaconda3/2022.05
conda activate
conda create --name tmptmp python=3.8
conda activate tmptmp
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install --upgrade pip setuptools wheel
cd Pointnet2_PyTorch && pip install -r requirements.txt
cd ..
pip install -r requirements.txt
pip install mayavi configobj

echo '=====testing gpu====='
python -c "import torch; x=torch.tensor(3, device='cuda'); print(x.device); print(torch.__version__);import trimesh"

echo '=====training======='
python3 train.py  --arch gan  --dataset_root_folder shapenet_models/da2_dataset/  --num_grasps_per_object 128 --niter 1000 --niter_decay 10000 --save_epoch_freq 50 --save_latest_freq 250 --run_test_freq 10 --num_threads 8 --dual_grasp True

conda deactivate
conda remove --name tmptmp --all
