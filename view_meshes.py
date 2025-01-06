from helper_functions import *



grasp_root_dir = "../DA2/data/da2_dataset/grasp"
mesh_root_dir = "../DA2/data/da2_dataset/simplified"

try:
    previous_grasp_files = np.load("grasp_files_to_save.npy")
    grasp_files_to_save = previous_grasp_files.tolist()
    grasp_files_scanned = np.load("grasp_files_scanned.npy").tolist()
except:
    print("No previous grasp files found")
    grasp_files_to_save = []
    grasp_files_scanned = []
print(len(grasp_files_to_save))
obj_counter = 0
for filename in os.listdir(grasp_root_dir):
    grasp_file_path = os.path.join(grasp_root_dir, filename)
    if os.path.isfile(grasp_file_path) and (filename not in grasp_files_to_save or filename not in grasp_files_scanned):
        # grasp_transforms, qua_for, qua_dex, qua_tor = load_dual_grasps(grasp_file_path)
        #Check if there are multiple objects in the same file
        show_obj(grasp_file_path, mesh_root_dir)
        obj_counter += 1
        text = input("Save the object? (press q to save, any other key to skip)")
        # while input != "q" and input != "n":
        #     text = input("Save the object? (press q to save, any other key to skip)")
        if text == "q":
            #Not saved
            #remove the directory
            filename = filename.replace(grasp_root_dir + "/", "")
            grasp_files_to_save.append(filename)
            print(filename + " saved")
        else:
            grasp_files_scanned.append(filename)
            print("Not saved")
        print("Objects scanned: " + str(obj_counter))
        if text == "n":
            print("quitting early...")
            break



#save the grasp files to a file
# np.save("grasp_files_to_save.npy", grasp_files_to_save)
# np.save("grasp_files_scanned.npy", grasp_files_scanned)
### REMOVE THIS ONE
#666e9c3ff214f7a661c1d97b345c8391_0.1698553260483882.h5
# 366fded292e63dfcf8f4382b90f1d4c3_0.02618897589009358.h5