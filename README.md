
# Dual-Arm Implementation of 6-DoF GraspNet
This is an adaption of (https://github.com/jsll/pytorch_6dof-graspnet) for dual-arm object manipulation. The following instructions are from the original repository. Further down are the instructions to train the dual-arm implementation.
This repository also has a seperate branch for training the VAE and GAN with split encoder/decoder and generator respectively for left and right arm on the branch `split_architectures`.

This is a PyTorch implementation of [6-DoF
GraspNet](https://arxiv.org/abs/1905.10520). The original Tensorflow
implementation can be found here <https://github.com/NVlabs/6dof-graspnet>.

# License

The source code is released under [MIT License](LICENSE) and the trained weights are released under [CC-BY-NC-SA 2.0](TRAINED_MODEL_LICENSE).

## Installation

This code has been tested with python 3.6, PyTorch 1.4 and CUDA 10.0 on Ubuntu
18.04. (I personally use 3.8.19, pytorch 1.13 and CUDA 11.7) To install do

1) `pip3 install torch==1.13.0+cu117 torchvision==0.14.0+cu117 -f <https://download.pytorch.org/whl/torch_stable.html>`

2) Clone this repository: `git clone
   git@github.com:jsll/pytorch_6dof-graspnet.git`.

3) Clone pointnet++: `https://github.com/erikwijmans/Pointnet2_PyTorch`.

4) Run `cd Pointnet2_PyTorch && pip3 install -r requirements.txt`

5) `cd pytorch_6dof-graspnet`

6) Run `pip3 install -r requirements.txt` to install necessary python libraries.

7) (Optional) Download the trained models either by running `sh
   checkpoints/download_models.sh` or manually from [here](https://drive.google.com/file/d/1B0EeVlHbYBki__WszkbY8A3Za941K8QI/view?usp=sharing). Trained
   models are released under [CC-BY-NC-SA 2.0](TRAINED_MODEL_LICENSE).

## Disclaimer

The pre-trained models released in this repo are retrained from scratch and not converted from the original ones <https://github.com/NVlabs/6dof-graspnet> trained in Tensorflow. I tried to convert the Tensorflow models but with no luck. Although I trained the new models for a substantial amount of time on the same training data, no guarantees to their performance compared to the original work can be given.

## Updates

In the paper, the authors only used gradient-based refinement. Recently, they released a Metropolis-Hastings
sampling method which they found to give better results in shorter time. As a result, I keep the Metropolis-Hastings sampling as the default for the demo.

This repository also includes an improved grasp sampling network which was
proposed here <https://github.com/NVlabs/6dof-graspnet>. The new grasp sampling
network is trained with [Implicit Maximum Likelihood Estimation](https://arxiv.org/pdf/2004.03590.pdf).

## Dataset for single-arm grasping

### Get ShapeNet Models

Download the meshes with ids written in [shapenet_ids.txt](shapenet_ids.txt) from <https://www.shapenet.org/.> Some of the objects are in `ShapenetCore` and `ShapenetSem`.

### Prepare ShapeNet Models

1. Clone and build: <https://github.com/hjwdzh/Manifold>
2. Create a watertight mesh version assuming the object path is model.obj: `manifold model.obj temp.watertight.obj -s`
3. Simplify it: `simplify -i temp.watertight.obj -o model.obj -m -r 0.02`

### Download the dataset

The dataset can be downloaded from [here](https://drive.google.com/open?id=1GkFrkvpP-R1letnv6rt_WLSX80o43Jjm). The dataset has 3 folders:

1) `grasps` folder: contains all the grasps for each object.
2) `meshes` folder: has the folder for all the meshes used. Except `cylinder` and `box` the rest of the folders are empty and need to be populated by the downloaded meshes from shapenet.
3) `splits` folder: contains the train/test split for each of the categories.


## Training

To train the grasp sampler (vae or gan) or the evaluator with bare minimum configurations run:

```shell
python3 train.py  --arch {vae,gan,evaluator}  --dataset_root_folder $DATASET_ROOT_FOLDER
```

where the `$DATASET_ROOT_FOLDER` is the path to the dataset you downloaded.

To monitor the training, run `tensorboard --logdir checkpoints/` and click <http://localhost:6006/>.

For more training options run:

```shell
python3 train.py  --help
```

# Dual-Arm training and DA2 dataset

Similar structure, can be downloaded from [here](https://drive.google.com/file/d/1Gb247xnwxbiy2psliTbu5DjMAi7pbBzn/view?usp=sharing).:
1) `grasps` folder: contains all the grasps pairs for each object.
2) `meshes` folder: has the folder for all the meshes used, not organized by object type like with the shapenet data
3) `splits` folder: contains the train/test split for each of the categories.

These three folders should be present under `shapenet_models/da2_dataset_small/`

## Scripts used to adapt the DA2 dataset
The scripts with some accompanying instructions can be found by switching to the branch `helper_code`

## Training Dual grasp

To train the grasp sampler (vae) with a configuration that works on a gpu with 8GB vram run:

```shell
python3 train.py  --arch vae  --dataset_root_folder shapenet_models/da2_dataset/  --num_grasps_per_object 32 --niter 1000 --niter_decay 10000 --save_epoch_freq 50 --save_latest_freq 250 --run_test_freq 10 --dual_grasp True
```

To look at the performance after training run"

```shell
python3 -m demo.main --grasp_sampler_folder checkpoints/vae/1_obj_small_data_unmergerd_vae_lr_0002_bs_32_scale_1_npoints_128_radius_02_latent_size_2/ --refinement_method gradient --dual_grasp --train_data --dataset_root_folder shapenet_models/da2_dataset_small/ --grasp_evaluator_folder checkpoints/evaluator/5_obj_small_datas_02_evaluator_lr_0002_bs_640_scale_1_npoints_128_radius_02/ --num_objects_to_show 20 --object_scale 0.3 --num_grasp_samples 2500
```

## DA2 Dataset Visualisation Code Snippets

`models/grasp_net.py` `line 66-75` code snippet to visualize the input pc with grasps and corresponding control points.
Similar snippet is present on `line 175-191` in the backward pass to visualize the generated control points by the network

`models/grasp_net.py` sets the input for the data as well as performs the backward pass that calculates the losses. These loss functions called are `control_point_l1_loss()` and `kl_divergence()` that are present in the file `models/losses.py`
The file `models/networks.py` creates the VAE, so the encoder and decoder, as well as perform the forward pass that calls the `encode()` end `decode()` functions respectively.

The dataloader part is handled by the files `data/base_dataset.py` (here the scaling occurs) and `data/grasp_sampling_data.py` for the VAE and `data/grasp_evaluator_data.py` for the evaluator model. On `lines 91-92` in `grasp_sampling.py` the grasp poses get flattened and those are used as input for the model together with the PC.

The file `utils/utils.py` contains multiple functions, most importantly the `get_control_point_tensor()` and `transform_control_points()` which transforms the sampled quaternions and translations into control points.


## Running in Habrok

Use the following command from your linux machine to ssh into Habrok GPU cluster

`ssh -X username@gpu1.hpc.rug.nl`

replace username with your student number (make sure you have an account). You login with your pass and third-party authenticator.

### Installation
Once in Habrok, navigate to the projects filesystem where you have more space

`cd /projects/sXXXXXX`

In my account I can simply `conda activate` every time I login, don't remember if I had to initialize it. If you can't just conda activate, you probably have to load the Anaconda module from Habrok module system, e.g.:

`module load Anaconda3/2022.05`

After you have conda available, you are ready to start installing.

```
conda create --name DA2graspnet python=3.8
conda activate DA2graspnet
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install --upgrade pip setuptools wheel
module load CUDA/11.7.0
git clone https://github.com/Anne-Jan/graspnetbackup.git
cd graspnetbackup
git clone https://github.com/erikwijmans/Pointnet2_PyTorch
cd Pointnet2_PyTorch && pip install -r requirements.txt
cd ..
```

Manually remove mayavi from requirements.txt and change `matplotlib` version to `3.1.3`. We will install latest mayavi version separately although not needed, just such that it doesnt complain during imports.

```
pip install -r requirements.txt
pip install mayavi #(no actual need just to not complain)
pip install configobj
```

### Running the demo
Since we are using headless server we cannot use mayavi. I have modified the demo to create the visualization in matplotlib and save it as a png file inside demo. Check `demo.main_headless.py` and `utils.visualization_utils_headless.py`. After you download weights and place in `checkpoints/` as in the original instructions, simply run:
`python -m demo.main_headless`

### Running training jobs
To submit a training job in Habrok, you must create a script e.g. `train_habrok_gpu.sh` that generally looks like this:

```
#!/bin/bash
#SBATCH --job-name="da2graspnet-train-example"
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=v100:1
#SBATCH --output=job-%j.log

cd /home1/sXXXXXX/graspnetbackup
module load CUDA/11.7.0
module load Anaconda3 
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

echo '======example usage======='
python -m demo.main_headless
```
This will allocate one v100 gpu for 3 hours. Check other available slurm params [here](https://wiki.hpc.rug.nl/habrok/job_management/scheduling_system)

Submit your job with:

`sbatch train_habrok_gpu.sh`

check your job status with:

`squeue -u sXXXXXX`

cancel a job with:

`scancel <JOBID>`

where you can read your `<JOBID>` from the command avove.



