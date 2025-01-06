import os
from helper_functions import *
import json

grasp_root_dir = "../DA2/data/grasps"
mesh_root_dir = "../DA2/data/"


#Do the transforms matter when feeding it to the network?
def countX(lst, x):
    count = 0
    for ele in lst:
        if (ele == x).all() :
            count = count + 1
    return count

for filename in os.listdir(grasp_root_dir):
    grasp_file_path = os.path.join(grasp_root_dir, filename)
    json_dict = json.load(open(grasp_file_path, "r"))
    transforms = np.array(json_dict["transforms"])
    #Only show the first 10 grasps
    transforms = transforms[:2]
    print(transforms.shape)
    obj_mesh = trimesh.load_mesh(os.path.join(mesh_root_dir, json_dict["object"]))
    obj_mesh.apply_transform(RigidTransform(np.eye(3), -obj_mesh.centroid).matrix)
    obj_scale = json_dict["object_scale"]
    obj_mesh = obj_mesh.apply_scale(obj_scale) 
    
    database = []
    wave = len(transforms)//3
    if wave == 0:
        wave = 1
    #Successful grasps are marked with a marker
    #Succesfull grasps to save are the transforms of the grasps
    successful_grasps = []
    marker = []
    for i, (t1, t2) in enumerate(transforms):
        
        current_t1 = countX(database, t1)
        current_t2 = countX(database, t2)
        color = i/wave*255
        code1 = color if color<=255 else 0
        code2 = color%255 if color>255 and color<=510 else 0
        code3 = color%510 if color>510 and color<=765 else 0
        successful_grasps.append((create_robotiq_marker(color=[code1, code2, code3]).apply_transform(t1), create_robotiq_marker(color=[code1, code2, code3]).apply_transform(t2)))

        trans1 = t1.dot(np.array([0,-0.067500/2-0.02*current_t1,0,1]).reshape(-1,1))[0:3]
        trans2 = t2.dot(np.array([0,-0.067500/2-0.02*current_t2,0,1]).reshape(-1,1))[0:3]

        tmp1 = trimesh.creation.icosphere(radius = 0.01).apply_transform(RigidTransform(np.eye(3), trans1).matrix)
        # tmp1 = trimesh.creation.icosphere(radius = 0.01)
        tmp1.visual.face_colors = [code1, code2, code3]
        tmp2 = trimesh.creation.icosphere(radius = 0.01).apply_transform(RigidTransform(np.eye(3), trans2).matrix)
        # tmp2 = trimesh.creation.icosphere(radius = 0.01)
        tmp2.visual.face_colors = [code1, code2, code3]
        marker.append(copy.deepcopy(tmp1))
        marker.append(copy.deepcopy(tmp2))
        database.append(t1)
        database.append(t2)
    
    #Show the object and the grasps
    # trimesh.Scene([obj_mesh]).show()
    trimesh.Scene([obj_mesh] + successful_grasps + marker).show()