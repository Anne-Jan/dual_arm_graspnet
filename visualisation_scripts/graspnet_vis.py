
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
import trimesh.scene
from utils.sample import Object
from renderer.online_object_renderer import OnlineObjectRenderer
from utils import utils
import trimesh.transformations as tra



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
        
        pc = apply_dropout(pc)
        pc = utils.regularize_pc_point_count(pc, npoints)
        pc_mean = np.mean(pc, 0, keepdims=True)
        pc[:, :3] -= pc_mean[:, :3]
        camera_pose[:3, 3] -= pc_mean[0, :3]

        return pc, camera_pose, in_camera_pose

if __name__ == '__main__':

    
    mesh_root_dir = 'shapenet/meshes/'
    grasp_root_dir = 'shapenet/grasps/'
    for filename in os.listdir(grasp_root_dir):
        print(grasp_root_dir)
        json_dict = json.load(open(grasp_root_dir + filename))
        object_model = Object(os.path.join(grasp_root_dir, json_dict['object']))
        object_model.rescale(json_dict['object_scale'])
        object_model = object_model.mesh
        object_mean = np.mean(object_model.vertices, 0, keepdims=1)
        object_model.vertices -= object_mean
        grasps = np.asarray(json_dict['transforms'])
        big_grasp = grasps[0]
        if len(grasps.shape) == 3:
            grasps[:, :3, 3] -= object_mean
        else:
            scale = 0.15
            json_dict['object_scale'] = json_dict['object_scale'] * scale
            scale= scale * 1.1
            S = np.diag([scale, scale, scale, 1])
            for i in range(len(grasps)):
               
                #Create the rotation matrix for the x axis
                R = tra.rotation_matrix(np.pi/2, [1, 0, 0])
                #rotate the grasp pc by 90 degrees about the x axis
                grasps[i][0][:3, :3] = np.matmul(grasps[i][0][:3, :3], R[:3, :3].T)
                grasps[i][1][:3, :3] = np.matmul(grasps[i][1][:3, :3], R[:3, :3].T)


                
                grasps[i][0] = S.dot(grasps[i][0])
                grasps[i][1] = S.dot(grasps[i][1])
                
      
        

        print(json_dict['object'])

        
        pc, camera_pose, _ = change_object_and_render(
                cad_path = json_dict['object'],
                cad_scale = json_dict['object_scale'])
        print(pc)
        for grasp in grasps:
            if len(grasp.shape) == 3:
                grasp[0] = camera_pose.dot(grasp[0])
                grasp[1] = camera_pose.dot(grasp[1])
            else:
                grasp = camera_pose.dot(grasp)
        if len(grasps.shape) == 4:
            grasps = grasps.reshape(-1, 4, 4)
        else:
            pc_pose = utils.inverse_transform(camera_pose)
            pc = pc.dot(pc_pose.T)
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
        
        # grasp_pc[2, 2] = 0.059
        # grasp_pc[3, 2] = 0.059
        # mid_point = 0.5 * (grasp_pc[2, :] + grasp_pc[3, :])

        # modified_grasp_pc = []
        # modified_grasp_pc.append(np.zeros((3, ), np.float32))
        # modified_grasp_pc.append(mid_point)
        # modified_grasp_pc.append(grasp_pc[2])
        # modified_grasp_pc.append(grasp_pc[4])
        # modified_grasp_pc.append(grasp_pc[2])
        # modified_grasp_pc.append(grasp_pc[3])
        # modified_grasp_pc.append(grasp_pc[5])

        # grasp_pc = np.asarray(modified_grasp_pc)
        # #rotate the grasp pc by 90 degrees about the x axis
        
        # # g = grasps[0]
        # g = big_grasp[0]
        # #set every point to the origin
        # pts = np.copy(grasp_pc)
        # pts -= np.expand_dims(g[:3, 3], 0)
        
        



       
        # # pts = np.matmul(grasp_pc, g[:3, :3].T)
        # # pts += np.expand_dims(g[:3, 3], 0)
        # # pts = grasp_pc
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
        #     m = mlab.plot3d(pts[:, 0],
        #                 pts[:, 1],
        #                 pts[:, 2],
        #                 color=gripper_color,
        #                 tube_radius=tube_radius,
        #                 opacity=1)
        #     print(m.type)
        #     # m.actor.actor.rotate_x(-90)
        # mlab.show()
        # break