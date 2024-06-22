
from __future__ import print_function
import mayavi.mlab as mlab


from utils.visualization_utils import *
# import mayavi.mlab as mlab
import time
import numpy as np
import trimesh
import copy
import json
import os
import torch
import trimesh
import trimesh.scene
from utils.sample import Object
from renderer.online_object_renderer import OnlineObjectRenderer
from utils import utils
import trimesh.transformations as tra
import utils.sample as sample
# from autolab_core import RigidTransform


# Function to render the mesh of the gripper (using robotiq here)
def create_robotiq_marker(color=[0, 0, 255], tube_radius=0.001, sections=6):
    """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.

    Args:
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
        sections (int, optional): Number of sections of each cylinder. Defaults to 6.

    Returns:
        trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
    """
    cfl = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[
            [4.10000000e-02, -7.27595772e-12, 0],
            [4.10000000e-02, -7.27595772e-12, 0.067500],
        ],
    )
    cfr = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[
            [-4.100000e-02, -7.27595772e-12, 0],
            [-4.100000e-02, -7.27595772e-12, 0.067500],
        ],
    )
    cb1 = trimesh.creation.cylinder(
        radius=0.002, sections=sections, segment=[[0, 0, -0.067500/2], [0, 0, 0]]
    )
    cb2 = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[[-0.085/2, 0, 0], [0.085/2, 0, 0]],
    )

    tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    tmp.visual.face_colors = color

    # z axis to x axis
    R = np.array([[1,0,0],[0,0,1],[0,-1,0]]).reshape(3,3)
    t =  np.array([0, 0, 0]).reshape(3,1)
    #
    T = np.r_[np.c_[R, t], [[0, 0, 0, 1]]]
    tmp.apply_transform(T)

    return tmp

def apply_dropout(pc):
        occlusion_nclusters = 0
        occlusion_dropout_rate = 0
        if occlusion_nclusters == 0 or occlusion_dropout_rate == 0.:
            return np.copy(pc)

        labels = utils.farthest_points(pc, occlusion_nclusters,
                                       utils.distance_by_translation_point)

        removed_labels = np.unique(labels)
        removed_labels = removed_labels[np.random.rand(removed_labels.shape[0])
                                        < occlusion_dropout_rate]
        if removed_labels.shape[0] == 0:
            return np.copy(pc)
        mask = np.ones(labels.shape, labels.dtype)
        for l in removed_labels:
            mask = np.logical_and(mask, labels != l)
        return pc[mask]

def change_object_and_render(
                                 cad_path,
                                 cad_scale,
                                 camera_pose=None,
                                 thread_id=0):
        all_poses = utils.nonuniform_quaternions()
        renderer = OnlineObjectRenderer(caching=True)
        npoints = 1024
        if camera_pose is None:
            viewing_index = np.random.randint(0, high=len(all_poses))
            camera_pose = all_poses[viewing_index]

        in_camera_pose = copy.deepcopy(camera_pose)

        _, _, pc, camera_pose = renderer.change_and_render(
            cad_path, cad_scale, in_camera_pose, thread_id)
        print(pc.shape)
        
        pc = apply_dropout(pc)
        pc = utils.regularize_pc_point_count(pc, npoints)
        pc_mean = np.mean(pc, 0, keepdims=True)
        pc[:, :3] -= pc_mean[:, :3]
        camera_pose[:3, 3] -= pc_mean[0, :3]

        return pc, camera_pose, in_camera_pose


def transform_to_pc_and_rotate(
                                 cad_path,
                                 cad_scale):
        
        npoints = 1024

        # obj = sample.Object(cad_path)
        # obj.rescale(cad_scale)
        # pc = obj.mesh.sample(npoints)
        obj = trimesh.load(cad_path)
        translation = -obj.centroid
        obj.apply_translation(translation)
        obj.apply_scale(cad_scale)
        pc = obj.sample(25000)
        pc = pc - np.mean(pc, axis=0)
        pc = apply_dropout(pc)
        pc = utils.regularize_pc_point_count(pc, npoints)
        pc_mean = np.mean(pc, 0, keepdims=True)
        pc[:, :3] -= pc_mean[:, :3]

        #Create the random rotation matrix
        R = tra.random_rotation_matrix()
        pc = np.matmul(pc, R[:3, :3].T)

        

        return pc, R

if __name__ == '__main__':

    
    mesh_root_dir = 'shapenet/meshes/'
    grasp_root_dir = 'shapenet/grasps/'
    for filename in os.listdir(grasp_root_dir):
        print(grasp_root_dir)
        #check if the file is a json file
        if not filename.endswith('.json'):
            continue
        json_dict = json.load(open(grasp_root_dir + filename))
        object_model = Object(os.path.join(grasp_root_dir, json_dict['object']))
        object_model.rescale(json_dict['object_scale'])
        object_model = object_model.mesh
        object_mean = np.mean(object_model.vertices, 0, keepdims=1)
        object_model.vertices -= object_mean
        grasps = np.asarray(json_dict['transforms'])
        big_grasp = grasps[0]



        successful_grasps = []
        if len(grasps.shape) == 3:
            grasps[:, :3, 3] -= object_mean
        else:
            # scale = 1
            scale = 0.2
            json_dict['object_scale'] = json_dict['object_scale'] * scale
            # scale= scale * 1.2
            
            S = np.diag([scale, scale, scale, 1])
            for i in range(len(grasps)):
                grasps[i][0] = S.dot(grasps[i][0])
                grasps[i][1] = S.dot(grasps[i][1])
                successful_grasps.append((create_robotiq_marker(color=[1, 1, 1]).apply_transform(grasps[i][0]), create_robotiq_marker(color=[1, 1, 1]).apply_transform(grasps[i][1])))
                #Create the rotation matrix for the x axis
                R = tra.rotation_matrix(np.pi/2, [1, 0, 0])
                #rotate the grasp pc by 90 degrees about the x axis
                grasps[i][0][:3, :3] = np.matmul(grasps[i][0][:3, :3], R[:3, :3].T)
                grasps[i][1][:3, :3] = np.matmul(grasps[i][1][:3, :3], R[:3, :3].T)


                
                
                
        # obj = trimesh.load(json_dict['object'])
        # #apply transform
        # # obj.vertices -= np.mean(obj.vertices, 0)  
        # translation = -obj.centroid
        # obj.apply_translation(translation)
        
        # obj.apply_scale(json_dict['object_scale'])
        # obj_pc = obj.sample(25000)
        # obj_pc = obj_pc - np.mean(obj_pc, axis=0)
        # obj_pc = apply_dropout(obj_pc)
        # obj_pc = utils.regularize_pc_point_count(obj_pc, 1024)
        # pc_mean = np.mean(obj_pc, 0, keepdims=True)
        # obj_pc[:, :3] -= pc_mean[:, :3]
        # obj_pc = trimesh.PointCloud(obj_pc)
        # obj_pc.vertices *= 0.1
        
        # trimesh.scene.Scene([obj_pc] + successful_grasps).show()

        
        # pc, camera_pose, _ = change_object_and_render(
        #         cad_path = json_dict['object'],
        #         cad_scale = json_dict['object_scale'])

        pc, R = transform_to_pc_and_rotate(
                cad_path = json_dict['object'],
                cad_scale = json_dict['object_scale'])
        
        for grasp in grasps:
            if len(grasp.shape) == 3:
                grasp[0] = R.dot(grasp[0])
                grasp[1] = R.dot(grasp[1])
            else:
                grasp = camera_pose.dot(grasp)
        if len(grasps.shape) == 4:
            print(grasps.shape)
            grasps = grasps.reshape(-1, 4, 4)
        # else:
        #     pc_pose = utils.inverse_transform(camera_pose)
        #     pc = pc.dot(pc_pose.T)
        pc = pc[:, :3]

        # visualize the scene with mayavi
        mlab.figure(bgcolor=(1, 1, 1))
        draw_scene(
                pc,
                grasps=grasps,
            )
        mlab.show()
        # break
        # grasp_pc = np.squeeze(utils.get_control_point_tensor(1, False), 0)

        # #swap two values of the grasp_pc that is 6x3
        # # grasp_pc[0], grasp_pc[1] = grasp_pc[1], grasp_pc[0]
        
        # grasp_pc[2, 2] = 0.059
        # grasp_pc[3, 2] = 0.059
        # mid_point = 0.5 * (grasp_pc[2, :] + grasp_pc[3, :])

        # #Shift all the points in the grasp pc towards points  0
        # zero_point = np.zeros((3, ), np.float32)
        # for point in grasp_pc:
        #     point[0] -= mid_point[0]
        #     point[1] -= mid_point[1]
        #     point[2] -= mid_point[2]
        # zero_point[0] -= mid_point[0]
        # zero_point[1] -= mid_point[1]
        # zero_point[2] -= mid_point[2]
        # mid_point = 0.5 * (grasp_pc[2, :] + grasp_pc[3, :])
        
        
        # modified_grasp_pc = []
        # modified_grasp_pc.append(zero_point)
        # modified_grasp_pc.append(mid_point)
        # modified_grasp_pc.append(grasp_pc[2])
        # modified_grasp_pc.append(grasp_pc[4])
        # modified_grasp_pc.append(grasp_pc[2])
        # modified_grasp_pc.append(grasp_pc[3])
        # modified_grasp_pc.append(grasp_pc[5])

        # grasp_pc = np.asarray(modified_grasp_pc)
        # #rotate the grasp pc by 90 degrees about the x axis
        
        # g = grasps[0]
        # g = big_grasp[0]
        # #set every point to the origin
        # pts = np.copy(grasp_pc)
        # pts -= np.expand_dims(g[:3, 3], 0)
        
        



       
        # pts = np.matmul(grasp_pc, g[:3, :3].T)
        # pts += np.expand_dims(g[:3, 3], 0)
        # pts = grasp_pc
        # gripper_color = (0.0, 1.0, 0.0)
        # if isinstance(gripper_color, list):
        #     m = mlab.plot3d(pts[:, 0],
        #                 pts[:, 1],
        #                 pts[:, 2],
        #                 color=gripper_color[i],
        #                 tube_radius=0.003,
        #                 opacity=1)
        #     # m.actor.actor.rotate_x(90)
        #     # print('rotated')
        # else:
        #     tube_radius = 0.001
        #     m = mlab.points3d(pts[:, 0],
        #                 pts[:, 1],
        #                 pts[:, 2],
        #                 color=gripper_color,
        #                 opacity=1)
        #     print(m.type)
        #     # m.actor.actor.rotate_x(-90)
        # mlab.show()
        # break