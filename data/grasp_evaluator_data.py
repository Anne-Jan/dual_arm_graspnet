import os
import torch
from data.base_dataset import BaseDataset, NoPositiveGraspsException
import numpy as np
from utils import utils
import random
import time
try:
    from Queue import Queue
except:
    from queue import Queue


class GraspEvaluatorData(BaseDataset):
    def __init__(self, opt, ratio_positive=0.3, ratio_hardnegative=0.5):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(
            opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataset_root_folder
        self.paths = self.make_dataset()
        self.size = len(self.paths)
        self.collision_hard_neg_queue = {}
        #self.get_mean_std()
        opt.input_nc = self.ninput_channels
        # ratio_positive = 0.5
        # ratio_hardnegative = 0.5
        self.ratio_positive = self.set_ratios(ratio_positive)
        self.ratio_hardnegative = self.set_ratios(ratio_hardnegative)
        # print('ratio_positive', self.ratio_positive, 'ratio_hardnegative', self.ratio_hardnegative)
        # self.ratio_hardnegative = 1.0
        # self.collision_hard_neg_num_perturbations = 5
        # print(self.collision_hard_neg_num_perturbations)

    def set_ratios(self, ratio):
        if int(self.opt.num_grasps_per_object * ratio) == 0:
            return 1 / self.opt.num_grasps_per_object
        return ratio

    def __getitem__(self, index):
        path = self.paths[index]
        if self.opt.balanced_data:
            data = self.get_uniform_evaluator_data(path)
        else:
            data = self.get_nonuniform_evaluator_data(path)
        gt_control_points = utils.transform_control_points_numpy(
            data[1], self.opt.num_grasps_per_object, mode='rt')
        # print(data[8].shape)
        all_gt_control_points = utils.transform_control_points_numpy(
            data[8], len(data[8]), mode='rt')


        
        
        # meta1 = data[8]
        meta = {}
        # meta['pc'] = meta1['pc']
        # meta['grasp_rt'] = meta1['grasp_rt']
        # meta['labels'] = meta1['labels']
        # meta['quality'] = meta1['quality']
        # meta['pc_pose'] = meta1['pc_pose']
        # meta['cad_path'] = meta1['cad_path']
        # meta['cad_scale'] = meta1['cad_scale']
        # meta['og_grasps'] = meta1['og_grasps']
        

        meta['pc'] = data[0][:, :, :3]
        if len(gt_control_points.shape) == 4:
            meta['grasp_rt'] = gt_control_points[:, :, :, :3]
        else:
            meta['grasp_rt'] = gt_control_points[:, :, :3]
        meta['labels'] = data[2]
        # print(meta['labels'])
        meta['quality'] = data[3]
        meta['pc_pose'] = data[4]
        meta['cad_path'] = data[5]
        meta['cad_scale'] = data[6]
        meta['all_target_grasps'] = all_gt_control_points[:, :, :, :3]
        # print(meta['all_target_grasps'].shape)
        # meta['og_grasps'] = data[7]
        # #First ones that are good
        # print(len(data[1]   ))
        # print(data[2])
        meta['good_og_grasps'] = data[1][:9,:,:,:]
        #Later ones that are bad, sometimes very bad
        meta['bad_og_grasps'] = data[1][9:,:,:,:]
        #shuffle the labels and the grasp rt to make sure that the network is not overfitting
        shuffle_idx = np.random.permutation(len(meta['labels']))
        meta['labels'] = meta['labels'][shuffle_idx]
        meta['grasp_rt'] = meta['grasp_rt'][shuffle_idx]
        meta['quality'] = meta['quality'][shuffle_idx]
        # print(meta['labels'])
        #print num of unique grasps
        # print("num_unique_pos_grasps", len(np.unique(meta['good_og_grasps'], axis=0)))
        # print("num_unique_neg_grasps", len(np.unique(meta['bad_og_grasps'], axis=0)))
        meta['og_grasps'] = data[1]
        # reshape to 64 x 4 x 4 from 32 x 1 x 2 x 4 x 4        
        if len(meta['og_grasps'].shape) == 5:
            print('reshaping')
            meta['og_grasps'] = meta['og_grasps'].reshape(-1, 4, 4)

        return meta

    def __len__(self):
        return self.size

    def get_uniform_evaluator_data(self, path, verify_grasps=False):
        pos_grasps, pos_qualities, neg_grasps, neg_qualities, obj_mesh, cad_path, cad_scale = self.read_grasp_file(
            path)

        output_pcs = []
        output_grasps = []
        output_qualities = []
        output_labels = []
        output_pc_poses = []
        output_cad_paths = [cad_path] * self.opt.batch_size
        output_cad_scales = np.asarray([cad_scale] * self.opt.batch_size,
                                       np.float32)

        num_positive = int(self.opt.batch_size * self.opt.ratio_positive)
        positive_clusters = self.sample_grasp_indexes(num_positive, pos_grasps,
                                                      pos_qualities)
        num_hard_negative = int(self.opt.batch_size *
                                self.opt.ratio_hardnegative)
        num_flex_negative = self.opt.batch_size - num_positive - num_hard_negative
        negative_clusters = self.sample_grasp_indexes(num_flex_negative,
                                                      neg_grasps,
                                                      neg_qualities)
        hard_neg_candidates = []
        # Fill in Positive Examples.

        for clusters, grasps, qualities in zip(
            [positive_clusters, negative_clusters], [pos_grasps, neg_grasps],
            [pos_qualities, neg_qualities]):
            for cluster in clusters:
                selected_grasp = grasps[cluster[0]][cluster[1]]
                selected_quality = qualities[cluster[0]][cluster[1]]
                hard_neg_candidates += utils.perturb_grasp(
                    selected_grasp,
                    self.collision_hard_neg_num_perturbations,
                    self.collision_hard_neg_min_translation,
                    self.collision_hard_neg_max_translation,
                    self.collision_hard_neg_min_rotation,
                    self.collision_hard_neg_max_rotation,
                )
        
        if verify_grasps:
            collisions, heuristic_qualities = utils.evaluate_grasps(
                output_grasps, obj_mesh)
            for computed_quality, expected_quality, g in zip(
                    heuristic_qualities, output_qualities, output_grasps):
                err = abs(computed_quality - expected_quality)
                if err > 1e-3:
                    raise ValueError(
                        'Heuristic does not match with the values from data generation {}!={}'
                        .format(computed_quality, expected_quality))

        # If queue does not have enough data, fill it up with hard negative examples from the positives.
        if path not in self.collision_hard_neg_queue or len(
                self.collision_hard_neg_queue[path]) < num_hard_negative:
            if path not in self.collision_hard_neg_queue:
                self.collision_hard_neg_queue[path] = []
            #hard negatives are perturbations of correct grasps.
            collisions, heuristic_qualities = utils.evaluate_grasps(
                hard_neg_candidates, obj_mesh)

            hard_neg_mask = collisions | (heuristic_qualities < 0.001)
            hard_neg_indexes = np.where(hard_neg_mask)[0].tolist()
            np.random.shuffle(hard_neg_indexes)
            for index in hard_neg_indexes:
                self.collision_hard_neg_queue[path].append(
                    (hard_neg_candidates[index], -1.0))
            random.shuffle(self.collision_hard_neg_queue[path])

        # Adding positive grasps
        for positive_cluster in positive_clusters:
            #print(positive_cluster)
            selected_grasp = pos_grasps[positive_cluster[0]][
                positive_cluster[1]]
            selected_quality = pos_qualities[positive_cluster[0]][
                positive_cluster[1]]
            output_grasps.append(selected_grasp)
            output_qualities.append(selected_quality)
            output_labels.append(1)

        # Adding hard neg
        for i in range(num_hard_negative):
            grasp, quality = self.collision_hard_neg_queue[path][i]
            output_grasps.append(grasp)
            output_qualities.append(quality)
            output_labels.append(0)

        self.collision_hard_neg_queue[path] = self.collision_hard_neg_queue[
            path][num_hard_negative:]
        # Adding flex neg
        if len(negative_clusters) != num_flex_negative:
            raise ValueError(
                'negative clusters should have the same length as num_flex_negative {} != {}'
                .format(len(negative_clusters), num_flex_negative))

        for negative_cluster in negative_clusters:
            selected_grasp = neg_grasps[negative_cluster[0]][
                negative_cluster[1]]
            selected_quality = neg_qualities[negative_cluster[0]][
                negative_cluster[1]]
            output_grasps.append(selected_grasp)
            output_qualities.append(selected_quality)
            output_labels.append(0)

        #self.change_object(cad_path, cad_scale)
        for iter in range(self.opt.num_grasps_per_object):
            if iter > 0:
                output_pcs.append(np.copy(output_pcs[0]))
                output_pc_poses.append(np.copy(output_pc_poses[0]))
            else:
                # pc, camera_pose, _ = self.change_object_and_render(
                #     cad_path,
                #     cad_scale,
                #     thread_id=torch.utils.data.get_worker_info().id
                #     if torch.utils.data.get_worker_info() else 0)
                pc, camera_pose = self.transform_to_pc_and_rotate(cad_path, cad_scale)
                output_pcs.append(pc)
                output_pc_poses.append(utils.inverse_transform(camera_pose))

            output_grasps[iter] = camera_pose.dot(output_grasps[iter])

        output_pcs = np.asarray(output_pcs, dtype=np.float32)
        output_grasps = np.asarray(output_grasps, dtype=np.float32)
        output_labels = np.asarray(output_labels, dtype=np.int32)
        output_qualities = np.asarray(output_qualities, dtype=np.float32)
        output_pc_poses = np.asarray(output_pc_poses, dtype=np.float32)

        return output_pcs, output_grasps, output_labels, output_qualities, output_pc_poses, output_cad_paths, output_cad_scales

    def get_nonuniform_evaluator_data(self, path, verify_grasps=False):
        
        # print(self.ratio_hardnegative,self.collision_hard_neg_max_translation, self.collision_hard_neg_min_translation, self.collision_hard_neg_max_rotation, self.collision_hard_neg_min_rotation, self.collision_hard_neg_num_perturbations)

        pos_grasps, pos_qualities, neg_grasps, neg_qualities, obj_mesh, cad_path, cad_scale, all_grasps = self.read_grasp_file(
            path)
        all_grasps = all_grasps.copy()
        output_pcs = []
        output_grasps = []
        output_qualities = []
        output_labels = []
        output_pc_poses = []
        output_cad_paths = [cad_path] * self.opt.num_grasps_per_object
        output_cad_scales = np.asarray(
            [cad_scale] * self.opt.num_grasps_per_object, np.float32)

        num_positive = int(self.opt.num_grasps_per_object *
                           self.ratio_positive)
        #only take the first 9 positive grasps
        positive_clusters = self.sample_grasp_indexes(num_positive, pos_grasps,
                                                      pos_qualities)
        num_negative = self.opt.num_grasps_per_object - num_positive

        negative_clusters = self.sample_grasp_indexes(num_negative, neg_grasps,
                                                      neg_qualities)
        hard_neg_candidates = []
        positive_grasps = []
        # Fill in Positive Examples.
        for positive_cluster in positive_clusters:
            selected_grasp = pos_grasps[positive_cluster[0]][
                positive_cluster[1]]
            selected_quality = pos_qualities[positive_cluster[0]][
                positive_cluster[1]]
            output_grasps.append(np.copy(selected_grasp))
            output_qualities.append(np.copy(selected_quality))
            output_labels.append(1)
            positive_grasps.append(selected_grasp)
            # print(self.collision_hard_neg_num_perturbations)
        
        for negative_cluster in negative_clusters:
            selected_grasp = neg_grasps[negative_cluster[0]][
                negative_cluster[1]]
            selected_quality = neg_qualities[negative_cluster[0]][
                negative_cluster[1]]
            hard_neg_candidates += utils.perturb_grasp(
                np.copy(selected_grasp),
                self.collision_hard_neg_num_perturbations,
                self.collision_hard_neg_min_translation,
                self.collision_hard_neg_max_translation,
                self.collision_hard_neg_min_rotation,
                self.collision_hard_neg_max_rotation,
            )
        # print("hard neg",len(hard_neg_candidates))
        if verify_grasps:
            collisions, heuristic_qualities = utils.evaluate_grasps(
                output_grasps, obj_mesh)
            for computed_quality, expected_quality, g in zip(
                    heuristic_qualities, output_qualities, output_grasps):
                err = abs(computed_quality - expected_quality)
                if err > 1e-3:
                    raise ValueError(
                        'Heuristic does not match with the values from data generation {}!={}'
                        .format(computed_quality, expected_quality))

        # If queue does not have enough data, fill it up with hard negative examples from the positives.
        if path not in self.collision_hard_neg_queue or self.collision_hard_neg_queue[
                path].qsize() < num_negative:
            if path not in self.collision_hard_neg_queue:
                self.collision_hard_neg_queue[path] = Queue()
            #hard negatives are perturbations of correct grasps.
            random_selector = np.random.rand()
            if random_selector <= self.ratio_hardnegative:
                # print('add hard neg')
                collisions, heuristic_qualities = utils.evaluate_grasps(
                    hard_neg_candidates, obj_mesh)
                hard_neg_mask = collisions | (heuristic_qualities < 0.001)
                hard_neg_indexes = np.where(hard_neg_mask)[0].tolist()
                np.random.shuffle(hard_neg_indexes)
                for index in hard_neg_indexes:
                    self.collision_hard_neg_queue[path].put(
                        (hard_neg_candidates[index], -1.0))
            if random_selector >= self.ratio_hardnegative or self.collision_hard_neg_queue[
                    path].qsize() < num_negative:
                # print('add flex neg')
                for negative_cluster in negative_clusters:
                    selected_grasp = neg_grasps[negative_cluster[0]][
                        negative_cluster[1]]
                    selected_quality = neg_qualities[negative_cluster[0]][
                        negative_cluster[1]]
                    self.collision_hard_neg_queue[path].put(
                        (selected_grasp, selected_quality))
        negative_grasps = []
        for idx in range(num_negative):
            #pick a random value between 0 and ratio_hardnegative + ratio_flexnegative
            ratio_flex_negative =  (1 - self.ratio_hardnegative - self.ratio_positive)
            #Make random float value between 0 and ratio_hardnegative + ratio_flexnegative
            random_selector = np.random.uniform(0, ratio_flex_negative + self.ratio_hardnegative)
            # print('qsize = ', self._collision_hard_neg_queue[path].qsize())
            # print(ratio_flex_negative)
            if random_selector <= self.ratio_hardnegative:
                # print('get hard neg')
                grasp, quality = self.collision_hard_neg_queue[path].get()
                grasp = hard_neg_candidates[idx]
                output_grasps.append(grasp)
                output_qualities.append(quality)
                output_labels.append(0)
                negative_grasps.append(grasp)
            else:
                # print('get flex neg')
                grasp = neg_grasps[negative_clusters[idx][0]][
                    negative_clusters[idx][1]]
                quality = neg_qualities[negative_clusters[idx][0]][
                    negative_clusters[idx][1]]
                output_grasps.append(grasp)
                output_qualities.append(quality)
                output_labels.append(0)
                negative_grasps.append(grasp)
        for iter in range(self.opt.num_grasps_per_object):
            if iter > 0:
                output_pcs.append(np.copy(output_pcs[0]))
                # output_pc_poses.append(0)
                output_pc_poses.append(np.copy(output_pc_poses[0]))
            else:
                # pc, camera_pose, _ = self.change_object_and_render(
                #     cad_path,
                #     cad_scale,
                #     thread_id=torch.utils.data.get_worker_info().id
                #     if torch.utils.data.get_worker_info() else 0)
                pc, camera_pose = self.transform_to_pc_and_rotate(cad_path, cad_scale)
                #self.change_object(cad_path, cad_scale)
                #pc, camera_pose, _ = self.render_random_scene()
                output_pcs.append(pc)
                output_pc_poses.append(utils.inverse_transform(camera_pose))
            if len(np.array(output_grasps).shape) == 4:
                output_grasps[iter][0] = camera_pose.dot(output_grasps[iter][0])
                output_grasps[iter][1] = camera_pose.dot(output_grasps[iter][1])
            else:
                output_grasps[iter] = camera_pose.dot(output_grasps[iter])

        for idx in range(len(all_grasps)):
            all_grasps[idx][0] = camera_pose.dot(all_grasps[idx][0])
            all_grasps[idx][1] = camera_pose.dot(all_grasps[idx][1])
        # print(self.collision_hard_neg_queue[
        #         path].qsize())
        output_pcs = np.asarray(output_pcs, dtype=np.float32)
        output_grasps = np.asarray(output_grasps, dtype=np.float32)
        output_labels = np.asarray(output_labels, dtype=np.int32)
        output_qualities = np.asarray(output_qualities, dtype=np.float32)
        output_pc_poses = np.asarray(output_pc_poses, dtype=np.float32)
        # print(np.array(positive_grasps).shape)
        return output_pcs, output_grasps, output_labels, output_qualities, output_pc_poses, output_cad_paths, output_cad_scales, np.array(positive_grasps), np.array(all_grasps)#, meta#, np.array(positive_grasps)
