from helper_functions import *



grasp_root_dir = "../DA2/data/grasp"

#Go through all the meshes and check if they are in the grasp directory
#If they are not, remove the mesh
grasps_to_remove = np.load("grasp_files_to_save.npy")
for filename in os.listdir(grasp_root_dir):
    if filename in grasps_to_remove:
        continue
    else:
        grasp_file_path = os.path.join(grasp_root_dir, filename)
        os.remove(grasp_file_path)
        print(filename + " removed")
#remove the grasps from the directory
# for filename in grasps_to_remove:
#     if filename in os.listdir(grasp_root_dir):
#         continue
#     else:
#     grasp_file_path = os.path.join(grasp_root_dir, filename)
#     os.remove(grasp_file_path)
#     print(filename + " removed")
