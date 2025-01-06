from helper_functions import *

grasp_root_dir = "../DA2/data/grasp"
mesh_root_dir = "../DA2/data/simplified"
successful_grasps_root_dir = "../DA2/data/grasps/"

#Remove files with different objects in one .obj file
#implement the pointnet pipeline
#check vgn
#generate grasps
#There are 4143 grasp files, segmentation proves difficult ask next meeting

num_to_save = 250
qua_thresh = 0.8 # set quality threshold for grasps
n_grasps_to_save = 1000# max num grasps to show
min_batch_size = 128
metric = "combined" # metric to filter grasps by: dex, for, tor, combined
count = 0
filecount = 0  # count of files processed
lowest_pos_grasps = 999999
procces_to_json = True
if metric == "combined":
    qua_thresh = qua_thresh * 3
for filename in os.listdir(grasp_root_dir):
    #Check if the file is already processe
    num_to_save_current = num_to_save
    filecount += 1
    grasp_file_path = os.path.join(grasp_root_dir, filename)
    if procces_to_json:
        # #Remove .h5 extension and replace with .json
        filename_json = filename.replace(".h5", ".json")
        # print(filename_json)
        if filename_json in os.listdir(successful_grasps_root_dir):
            print(f'{filename_json} already exists, skipping')
            continue
    if os.path.isfile(grasp_file_path):
        grasp_transforms, qua_for, qua_dex, qua_tor = load_dual_grasps(grasp_file_path)
        # logic to visualize mesh + gripper mesh for loaded grasps, potentially filtered by quality threshold
        print(f'Processing {grasp_file_path}, {filecount} files processed')
        
        #Check if there are multiple objects in the same file
        # count_objects_in_pc(grasp_file_path, mesh_root_dir)
        obj_pc, obj_mesh, successful_grasps, pos_metric, marker, successful_grasps_to_save = filter_grasps_based_on_metric_partial(grasp_file_path, metric, qua_thresh, n_grasps = n_grasps_to_save, mesh_root_dir = mesh_root_dir)
        #only take an equal number of negative grasps
        neg_obj_pc, neg_obj_mesh, neg_grasps, neg_metric, neg_marker, neg_grasps_to_save = filter_negative_grasps(grasp_file_path, metric, n_grasps = len(successful_grasps), mesh_root_dir = mesh_root_dir)
        if len(successful_grasps) < lowest_pos_grasps:
            lowest_pos_grasps = len(successful_grasps)
            #Change later, the min batch size is 128
            if len(successful_grasps) < min_batch_size:
                continue
            print(f'Lowest number of positive grasps found: {lowest_pos_grasps}')
        if len(successful_grasps) == 0:
            print(f'No grasps with {metric} metric above {qua_thresh} go to next file')
            continue
        elif len(successful_grasps) < n_grasps_to_save:
            print(f'Only {len(successful_grasps)} grasps with {metric} metric above {qua_thresh} found, showing all of them and skipping this object')
            # continue
            n_found_grasps = len(successful_grasps)

        elif len(successful_grasps) != len(neg_grasps):
            print("number of positive and negative grasps do not match")
            continue
        else:
            print(f'All positve and negative = 2 X {n_grasps_to_save} have been found')
            n_found_grasps = (len(successful_grasps) + len(neg_grasps))
        
        # visualize sucesful grasps  
        # show_scene_with_grasps(n_found_grasps, metric, qua_thresh, obj_pc, obj_mesh, successful_grasps, marker, "") 
        # visualize negative grasps
        # show_scene_with_grasps(n_found_grasps, metric, qua_thresh, neg_obj_pc, neg_obj_mesh, neg_grasps, neg_marker, "")    
        #Save the object point cloud and grasps to a file
        #combine the negative and positive grasps
        grasps_to_save = np.concatenate((neg_grasps_to_save, successful_grasps_to_save), axis = 0)
        # grasps_to_save = len(neg_grasps_to_save) +len(successful_grasps_to_save)
        #Check if the pos_metric has the shape (n,) not (n,1)
        if len(np.array(pos_metric).shape) == 2:            
            pos_metric = pos_metric.reshape(-1)
        metric_to_save = np.concatenate((neg_metric, pos_metric), axis = 0)
        print(len(grasps_to_save),len(metric_to_save))
        save_to_json(grasp_file_path, successful_grasps_root_dir, grasps_to_save, metric_to_save)
        num_to_save_current -= filecount
        print(f'Number of files left to save: {num_to_save_current}')
        if num_to_save_current == 0:
            break
