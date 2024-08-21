import os
import torch
from data.base_dataset import BaseDataset, NoPositiveGraspsException
import numpy as np
from utils import utils
from utils.visualization_utils import *


class GraspSamplingData(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(
            opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataset_root_folder
        self.paths = self.make_dataset()
        self.size = len(self.paths)
        #self.get_mean_std()
        opt.input_nc = self.ninput_channels
        self.i = 0

    def __getitem__(self, index):
        path = self.paths[index]
        pos_grasps, pos_qualities, _, _, _, cad_path, cad_scale = self.read_grasp_file(
            path)
        meta = {}
        try:
            all_clusters = self.sample_grasp_indexes(
                self.opt.num_grasps_per_object, pos_grasps, pos_qualities)
        except NoPositiveGraspsException:
            if self.opt.skip_error:
                return None
            else:
                return self.__getitem__(np.random.randint(0, self.size))

        #self.change_object(cad_path, cad_scale)
        # pc, camera_pose, _ = self.render_random_scene()

        # pc, camera_pose, _ = self.change_object_and_render(
        #     cad_path,
        #     cad_scale,
        #     thread_id=torch.utils.data.get_worker_info().id
        #     if torch.utils.data.get_worker_info() else 0)
        #Check type of pc
        # print(pc[0].shape)
        
        pc, camera_pose = self.transform_to_pc_and_rotate(cad_path, cad_scale)
        
        output_qualities = []
        output_grasps = []
        # print(pos_qualities)
        for iter in range(self.opt.num_grasps_per_object):
            selected_grasp_index = all_clusters[iter]

            selected_grasp = pos_grasps[selected_grasp_index[0]][
                selected_grasp_index[1]]
            selected_quality = pos_qualities[selected_grasp_index[0]][
                selected_grasp_index[1]]
            output_qualities.append(selected_quality)
            #Dot product with camera pose, differentiate between 1 and 2 grasps
            if len(selected_grasp.shape) == 3:
                selected_grasp[0] = camera_pose.dot(selected_grasp[0])
                selected_grasp[1] = camera_pose.dot(selected_grasp[1])
                output_grasps.append(selected_grasp)
            else:
                output_grasps.append(camera_pose.dot(selected_grasp))
        gt_control_points = utils.transform_control_points_numpy(
            np.array(output_grasps), self.opt.num_grasps_per_object, mode='rt')
        
        meta['pc'] = np.array([pc] * self.opt.num_grasps_per_object)[:, :, :3]
        # print(meta['pc'].shape)

        if len(np.array(output_grasps).shape) == 4:
            #reshape pos grasp to (num_grasps, 2, 4, 4)
            # pos_grasps = np.array(pos_grasps).reshape(
            #     len(pos_grasps), 2, 4, 4)
            #Reshape it to (num_grasps, 2, 16)
            meta['og_grasps'] = np.array(output_grasps)
            # print(np.array(output_grasps[0]).shape)
            # print("unflattened", np.array(output_grasps)[0])
            # print("flattened", np.array(output_grasps).reshape(len(output_grasps), -1)[0])
            # print(xd)
            #reshape it from (num_grasps, 2, 4, 4) to twice the number of grasps, 16
            # reshaped_grasps = np.array(output_grasps).reshape(64, 4, 4)
            # reshaped_grasps = reshaped_grasps.reshape(64, -1)
            # print("unflattened", np.array(output_grasps)[0])
            # print("flattened", reshaped_grasps[0], "flattened 2", reshaped_grasps[1])
            # meta['grasp_rt'] = reshaped_grasps

            
            meta['grasp_rt'] = np.array(output_grasps).reshape(
                len(output_grasps), -1)
            meta['target_cps'] = np.array(gt_control_points[:, :, :, :3])
            # meta['grasp_rt'] = np.array(output_grasps).reshape(
            #     len(output_grasps), 2, -1)
        else:
            meta['og_grasps'] = np.array(output_grasps)
            meta['grasp_rt'] = np.array(output_grasps).reshape(
                len(output_grasps), -1)
            meta['target_cps'] = np.array(gt_control_points[:, :, :3])
        # print(meta['grasp_rt'].shape)

        meta['pc_pose'] = np.array([utils.inverse_transform(camera_pose)] *
                                   self.opt.num_grasps_per_object)
        meta['cad_path'] = np.array([cad_path] *
                                    self.opt.num_grasps_per_object)
        meta['cad_scale'] = np.array([cad_scale] *
                                     self.opt.num_grasps_per_object)
        meta['quality'] = np.array(output_qualities)
        
        
        # print("Control points shape" + str(gt_control_points.shape))
        # print("targetcps" + str(meta['target_cps'].shape))
        return meta

    def __len__(self):
        return self.size