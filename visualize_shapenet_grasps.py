import json
import os
import trimesh
import numpy as np
from helper_functions import *

#read a json file

# json_path = "../shapenet_data/grasps/Bottle_3ae3a9b74f96fef28fe15648f042f0d9_0.0022323601.json"
json_path = "../shapenet_data/grasps/Bottle_3dbd66422997d234b811ffed11682339_0.0515355444.json"
# json_path = "../shapenet_data/grasps/box_box001_1.0.json"
json_dict = json.load(open(json_path))
quality='quality_flex_object_in_gripper'

#list the keys
print(json_dict.keys())
# print(json_dict['roll_angles'])
#
object_model = trimesh.load_mesh(os.path.join("../shapenet_data", json_dict['object']))
mesh_scale = json_dict['object_scale']
# object_model.show()
object_model.apply_transform(RigidTransform(np.eye(3), -object_model.centroid).matrix)
object_model = object_model.apply_scale(mesh_scale)
object_mean = np.mean(object_model.vertices, 0, keepdims=1)

object_model.vertices -= object_mean
grasps = np.asarray(json_dict['transforms'])
print(grasps.shape)
mesh_normals = np.asarray(json_dict['mesh_normals'])

grasps[:, :3, 3] -= object_mean
print(grasps.size)
flex_qualities = np.asarray(json_dict[quality])
try:
    heuristic_qualities = np.asarray(
        json_dict['quality_number_of_contacts'])
except KeyError:
    heuristic_qualities = np.ones(flex_qualities.shape)
successful_mask = np.logical_and(flex_qualities > 0.01,
                                    heuristic_qualities > 0.01)

positive_grasp_indexes = np.where(successful_mask)[0]
mesh_normals = mesh_normals[positive_grasp_indexes]
roll_angles = np.asarray(json_dict['roll_angles'])[positive_grasp_indexes]
negative_grasp_indexes = np.where(~successful_mask)[0]
positive_grasps = grasps[positive_grasp_indexes, :, :]
negative_grasps = grasps[negative_grasp_indexes, :, :]
positive_qualities = heuristic_qualities[positive_grasp_indexes]
negative_qualities = heuristic_qualities[negative_grasp_indexes]
print(len(positive_grasps))
print(len(mesh_normals))
n_grasps = 10
database = []
wave = n_grasps//3
if wave == 0:
    wave = 1
successful_grasps = []
marker = []
def countX(lst, x):
    count = 0
    for ele in lst:
        if (ele == x).all() :
            count = count + 1
    return count
for i, t1 in enumerate(positive_grasps):
        #In case we want only the best grasp, we need to check if it is visible
        #If not, we need to select the next best grasp and check again
        current_t1 = countX(database, t1)
        #Change the orientation of the grasp
        # # current_t2 = countX(database, t2)
        angle = roll_angles[i]
        color = i/wave*255
        code1 = color if color<=255 else 0
        code2 = color%255 if color>255 and color<=510 else 0
        code3 = color%510 if color>510 and color<=765 else 0
        successful_grasps.append(create_robotiq_marker(color=[code1, code2, code3]).apply_transform(t1))
        # successful_grasps.append(create_robotiq_marker(color=[code1, code2, code3]))
        # trans1 = t1.dot(np.array([0,-0.067500/2-0.02*current_t1,0,1]).reshape(-1,1))[0:3]
        trans1 = t1.dot(np.array([0,0,0,1]).reshape(-1,1))[0:3]

        tmp1 = trimesh.creation.icosphere(radius = 0.01).apply_transform(RigidTransform(np.eye(3), trans1).matrix)
        tmp1.visual.face_colors = [code1, code2, code3]
        marker.append(copy.deepcopy(tmp1))
        database.append(t1)
        if len(successful_grasps) == n_grasps:
            break

def show_scene_with_grasps(n_grasps, obj_mesh, successful_grasps, marker):
    # visualize sucesful grasps        
    print(f'Showing {n_grasps} grasps')
    # trimesh.Scene([obj_mesh] + successful_grasps).show()
    trimesh.Scene([obj_mesh] + successful_grasps + marker).show()
show_scene_with_grasps(n_grasps, object_model, successful_grasps, marker)