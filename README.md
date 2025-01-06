# Helper Functions

## To generate the dataset
Have the original data present obtained from the DA2 repo
Folder structure from this directory should be:
1. `../DA2/data/grasp/` for the grasp files
2. `../DA2/data/simplified/` for the meshes in obj file format
3. `../DA2/data/grasps/` for the create_dataset script to place the selected grasp per object
4. `../DA2/data/meshes/` for the convert_mesh_to_stl.py script to place the converted .stl files
5. `../DA2/data/splits/` for the create_train_test_split.py to place the create train/test split .json file
Run `python3 create_dataset.py`
Run `python3 convert_mesh_to_stl.py`
Run `python3 create_train_test_split.py`
Put the /meshes, /grasps and /splits folder in the unified_grasp_data/da2_dataset folder of the graspnet repo 

## Other helper functions
Run `create_grapshs.py` to generate the graphs for the work combined with the data from `model_loss_data/`  
Run `python3 view_meshes.py` to go over all the meshes in the DA2 dataset and select if you want them stored or not. The results are saved in a .npy file called `grasp_files_to_save.npy`.   
If you wish to quit halfway press `n` and all progress will be saved in `grasp_files_scanned.npy` and `grasp_files_to_save.npy`  
To remove all other meshes afterwards (the ones that were not saved) run `python3 remove_saved_meshes.py`.  
The file `visualize_shapenet_grasps.py` was used to visually inspect the meshes and grasps used to train the original 6DOF-GraspNet.  
The file `view_stl_and_grasps.py` was used to visually inspect the meshes and grasps from DA2  
 