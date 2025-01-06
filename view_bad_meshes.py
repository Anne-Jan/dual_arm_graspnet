from helper_functions import *

import numpy as np

filenames = np.load("grasp_files_to_remove.npy")
grasp_root_dir = "../DA2/data/da2_dataset/full"
mesh_root_dir = "../DA2/data/da2_dataset/simplified"
#pc_root_dir = "../DA2/data/pointclouds"

grasp_files = []
obj_counter = 0
for filename in filenames:
    grasp_file_path = os.path.join(grasp_root_dir, filename)
    if os.path.isfile(grasp_file_path):
        # grasp_transforms, qua_for, qua_dex, qua_tor = load_dual_grasps(grasp_file_path)
        #Check if there are multiple objects in the same file
        show_obj(grasp_file_path, mesh_root_dir)
        obj_counter += 1
        text = input("Save the object? (press q to save, any other key to skip)")
        if text == "q":
            #remove the directory
            filename = filename.replace(grasp_root_dir + "/", "")
            grasp_files.append(filename)
            print(filename + " saved")
        else:
            print("Not saved")
        print("Objects scanned: " + str(obj_counter))


#save the grasp files to a file
# np.save("grasp_files.npy", grasp_files)
# 7d9f29727e55d2b19eeed882992d9dd_0.014252784826702452.h5