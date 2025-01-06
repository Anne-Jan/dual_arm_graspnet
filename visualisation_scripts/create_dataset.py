from helper_functions import *

# grasp annotation for the chair object above
grasp_root_dir = "da2/grasp"
mesh_root_dir = "da2/simplified"

#Remove files with different objects in one .obj file
#implement the pointnet pipeline
#check vgn
#generate grasps
#There are 4143 grasp files, segmentation proves difficult ask next meeting

num_to_save = 1
qua_thresh = 0.8 # set quality threshold for grasps
n_grasps_to_save = 10# max num grasps to show
metric = "combined" # metric to filter grasps by: dex, for, tor, combined
count = 0
filecount = 0  # count of files processed
if metric == "combined":
    qua_thresh = qua_thresh * 3
for filename in os.listdir(grasp_root_dir):
    filecount += 1
    grasp_file_path = os.path.join(grasp_root_dir, filename)
    if os.path.isfile(grasp_file_path):
        grasp_transforms, qua_for, qua_dex, qua_tor = load_dual_grasps(grasp_file_path)
        # logic to visualize mesh + gripper mesh for loaded grasps, potentially filtered by quality threshold
        print(f'Processing {grasp_file_path}, {filecount} files processed')
        
        #Check if there are multiple objects in the same file
        # count_objects_in_pc(grasp_file_path, mesh_root_dir)
        neg_obj_pc, neg_obj_mesh, neg_grasps, neg_metric, neg_marker, neg_grasps_to_save = filter_negative_grasps(grasp_file_path, metric, n_grasps = n_grasps_to_save, mesh_root_dir = mesh_root_dir)
        obj_pc, obj_mesh, successful_grasps, pos_metric, marker, successful_grasps_to_save = filter_grasps_based_on_metric_partial(grasp_file_path, metric, qua_thresh, n_grasps = n_grasps_to_save, mesh_root_dir = mesh_root_dir)

        if len(successful_grasps) == 0:
            print(f'No grasps with {metric} metric above {qua_thresh} go to next file')
            continue
        elif len(successful_grasps) < n_grasps_to_save:
            print(f'Only {len(successful_grasps)} grasps with {metric} metric above {qua_thresh} found, showing all of them')
            continue
            n_found_grasps = len(successful_grasps)

        elif len(successful_grasps) != len(neg_grasps):
            print("number of positive and negative grasps do not match")
            continue
        else:
            print(f'All positve and negative = 2 X {n_grasps_to_save} have been found')
            n_found_grasps = (len(successful_grasps) + len(neg_grasps))
        
        # visualize sucesful grasps  
        
        successful_grasps = successful_grasps[:1]
        show_scene_with_grasps(n_found_grasps, metric, qua_thresh, obj_pc, obj_mesh, successful_grasps, marker, "") 
        # visualize negative grasps
        # show_scene_with_grasps(n_found_grasps, metric, qua_thresh, neg_obj_pc, neg_obj_mesh, neg_grasps, neg_marker, "")    
        #Save the object point cloud and grasps to a file
        #combine the negative and positive grasps
        grasps_to_save = np.concatenate((neg_grasps_to_save, successful_grasps_to_save), axis = 0)
        metric_to_save = neg_metric + pos_metric
        # print(len(grasps_to_save))
        # # save_to_json(grasp_file_path, successful_grasps_root_dir, grasps_to_save, metric_to_save)
        # num_to_save -= 1
        # print(f'Number of files left to save: {num_to_save}')
        # if num_to_save == 0:
        #     break
