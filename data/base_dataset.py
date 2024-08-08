import torch.utils.data as data
import numpy as np
import pickle
import os
import copy
import json
import torch
from utils.sample import Object
from utils import utils
from utils.visualization_utils import *

import glob
from renderer.online_object_renderer import OnlineObjectRenderer
import threading
import trimesh
import trimesh.transformations as tra

class NoPositiveGraspsException(Exception):
    """raised when there's no positive grasps for an object."""
    pass


class BaseDataset(data.Dataset):
    def __init__(self,
                 opt,
                 caching=True,
                 min_difference_allowed=(0, 0, 0),
                 max_difference_allowed=(3, 3, 0),
                 collision_hard_neg_min_translation=(-0.03, -0.03, -0.03),
                 collision_hard_neg_max_translation=(0.03, 0.03, 0.03),
                 collision_hard_neg_min_rotation=(-0.6, -0.2, -0.6),
                 collision_hard_neg_max_rotation=(+0.6, +0.2, +0.6),
                 collision_hard_neg_num_perturbations=10):
        super(BaseDataset, self).__init__()
        self.opt = opt
        self.mean = 0
        self.std = 1
        self.ninput_channels = None
        self.current_pc = None
        self.caching = caching
        self.cache = {}
        self.collision_hard_neg_min_translation = collision_hard_neg_min_translation
        self.collision_hard_neg_max_translation = collision_hard_neg_max_translation
        self.collision_hard_neg_min_rotation = collision_hard_neg_min_rotation
        self.collision_hard_neg_max_rotation = collision_hard_neg_max_rotation
        self.collision_hard_neg_num_perturbations = collision_hard_neg_num_perturbations
        self.lock = threading.Lock()
        for i in range(3):
            assert (collision_hard_neg_min_rotation[i] <=
                    collision_hard_neg_max_rotation[i])
            assert (collision_hard_neg_min_translation[i] <=
                    collision_hard_neg_max_translation[i])

        self.renderer = OnlineObjectRenderer(caching=True)

        if opt.use_uniform_quaternions:
            self.all_poses = utils.uniform_quaternions()
        else:
            self.all_poses = utils.nonuniform_quaternions()

        self.eval_files = [
            json.load(open(f)) for f in glob.glob(
                os.path.join(self.opt.dataset_root_folder, 'splits', '*.json'))
        ]

    def apply_dropout(self, pc):
        if self.opt.occlusion_nclusters == 0 or self.opt.occlusion_dropout_rate == 0.:
            return np.copy(pc)

        labels = utils.farthest_points(pc, self.opt.occlusion_nclusters,
                                       utils.distance_by_translation_point)

        removed_labels = np.unique(labels)
        removed_labels = removed_labels[np.random.rand(removed_labels.shape[0])
                                        < self.opt.occlusion_dropout_rate]
        if removed_labels.shape[0] == 0:
            return np.copy(pc)
        mask = np.ones(labels.shape, labels.dtype)
        for l in removed_labels:
            mask = np.logical_and(mask, labels != l)
        return pc[mask]

    def render_random_scene(self, camera_pose=None):
        """
          Renders a random view and return (pc, camera_pose, object_pose). 
          object_pose is None for single object per scene.
        """
        if camera_pose is None:
            viewing_index = np.random.randint(0, high=len(self.all_poses))
            camera_pose = self.all_poses[viewing_index]

        in_camera_pose = copy.deepcopy(camera_pose)
        _, _, pc, camera_pose = self.renderer.render(in_camera_pose)
        pc = self.apply_dropout(pc)
        pc = utils.regularize_pc_point_count(pc, self.opt.npoints)
        pc_mean = np.mean(pc, 0, keepdims=True)
        pc[:, :3] -= pc_mean[:, :3]
        camera_pose[:3, 3] -= pc_mean[0, :3]

        return pc, camera_pose, in_camera_pose
    
    def transform_to_pc_and_rotate(self,
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
        #Change each point from numpy.float64 to numpy.float32
        
        pc = pc - np.mean(pc, axis=0)
        pc = self.apply_dropout(pc)
        pc = utils.regularize_pc_point_count(pc, npoints)
        pc_mean = np.mean(pc, 0, keepdims=True)
        pc[:, :3] -= pc_mean[:, :3]

        #Create the random rotation matrix
        R = tra.random_rotation_matrix()
        pc = np.matmul(pc, R[:3, :3].T)
        pc = pc.astype(np.float32)
        

        return pc, R
    
    def change_object_and_render(self,
                                 cad_path,
                                 cad_scale,
                                 camera_pose=None,
                                 thread_id=0):
        if camera_pose is None:
            viewing_index = np.random.randint(0, high=len(self.all_poses))
            camera_pose = self.all_poses[viewing_index]

        in_camera_pose = copy.deepcopy(camera_pose)
        _, _, pc, camera_pose = self.renderer.change_and_render(
            cad_path, cad_scale, in_camera_pose, thread_id)
        pc = self.apply_dropout(pc)
        pc = utils.regularize_pc_point_count(pc, self.opt.npoints)
        pc_mean = np.mean(pc, 0, keepdims=True)
        pc[:, :3] -= pc_mean[:, :3]
        camera_pose[:3, 3] -= pc_mean[0, :3]

        return pc, camera_pose, in_camera_pose

    def change_object(self, cad_path, cad_scale):
        self.renderer.change_object(cad_path, cad_scale)

    def read_grasp_file(self, path, return_all_grasps=False):
        file_name = path
        if self.caching and file_name in self.cache:
            pos_grasps, pos_qualities, neg_grasps, neg_qualities, cad, cad_path, cad_scale = copy.deepcopy(
                self.cache[file_name])
            return pos_grasps, pos_qualities, neg_grasps, neg_qualities, cad, cad_path, cad_scale
        pos_grasps, pos_qualities, neg_grasps, neg_qualities, cad, cad_path, cad_scale = self.read_object_grasp_data(
            path,
            ratio_of_grasps_to_be_used=self.opt.grasps_ratio,
            return_all_grasps=return_all_grasps)

        if self.caching:
            self.cache[file_name] = (pos_grasps, pos_qualities, neg_grasps,
                                     neg_qualities, cad, cad_path, cad_scale)
            return copy.deepcopy(self.cache[file_name])
        return pos_grasps, pos_qualities, neg_grasps, neg_qualities, cad, cad_path, cad_scale

    def read_object_grasp_data(self,
                               json_path,
                               quality='quality_flex_object_in_gripper',
                               ratio_of_grasps_to_be_used=1.,
                               return_all_grasps=False):
        """
        Reads the grasps from the json path and loads the mesh and all the 
        grasps.
        """
        num_clusters = self.opt.num_grasp_clusters
        root_folder = self.opt.dataset_root_folder
        if num_clusters <= 0:
            raise NoPositiveGraspsException
        # print(json_path)
        json_dict = json.load(open(json_path))

        object_model = Object(os.path.join(root_folder, json_dict['object']))
        object_model.rescale(json_dict['object_scale'])
        object_model = object_model.mesh
        object_mean = np.mean(object_model.vertices, 0, keepdims=1)

        object_model.vertices -= object_mean
        grasps = np.asarray(json_dict['transforms'])
        if len(grasps.shape) == 4:
            #shift all grasps slightly along the z-axis
            # grasps[:, :, :3, 3] += np.array([0, 0, 0.05])

            # grasps[:, :, :3, 3] -= object_mean
            scale = 0.2
            json_dict['object_scale'] = json_dict['object_scale'] * scale
            # scale = 1.25 * scale
            S = np.diag([scale, scale, scale, 1])
            for i in range(len(grasps)):
                R = tra.rotation_matrix(np.pi/2, [1, 0, 0])
                #rotate the grasp pc by 90 degrees about the x axis
                grasps[i][0][:3, :3] = np.matmul(grasps[i][0][:3, :3], R[:3, :3].T)
                grasps[i][1][:3, :3] = np.matmul(grasps[i][1][:3, :3], R[:3, :3].T)
                
                grasps[i][0] = S.dot(grasps[i][0])
                grasps[i][1] = S.dot(grasps[i][1])

                def compute_center_point(grasp_matrix):
                    # Calculate the mean of the first three columns (x, y, z coordinates)
                    center_x = np.mean(grasp_matrix[:, 0])
                    center_y = np.mean(grasp_matrix[:, 1])
                    center_z = np.mean(grasp_matrix[:, 2])
                    return center_x, center_y, center_z

                def scale_and_translate_gripper(grasp_matrix, center_point, scaling_factor, translation_vector):
                    # Extract the center coordinates
                    C_x, C_y, C_z = center_point
                    
                    # Create a new matrix for the scaled and translated coordinates
                    scaled_translated_matrix = np.copy(grasp_matrix)
                    
                    # Apply the scaling transformation to the first three columns (x, y, z coordinates)
                    for i in range(grasp_matrix.shape[0]):
                        x, y, z = grasp_matrix[i, 0], grasp_matrix[i, 1], grasp_matrix[i, 2]
                        x_prime = C_x + scaling_factor * (x - C_x)
                        y_prime = C_y + scaling_factor * (y - C_y)
                        z_prime = C_z + scaling_factor * (z - C_z)
                        # Apply translation
                        x_prime += translation_vector[0]
                        y_prime += translation_vector[1]
                        z_prime += translation_vector[2]
                        scaled_translated_matrix[i, 0] = x_prime
                        scaled_translated_matrix[i, 1] = y_prime
                        scaled_translated_matrix[i, 2] = z_prime
                    
                    return scaled_translated_matrix
                scale_factor = 1.5
                translation_vector = [0, 0, 0]
                # Example grasp matrix (4x4 matrix)
                # grasps[i][0] = scale_and_translate_gripper(grasps[i][0], compute_center_point(grasps[i][0]), scale_factor, translation_vector)    
                # grasps[i][1] = scale_and_translate_gripper(grasps[i][1], compute_center_point(grasps[i][1]), scale_factor, translation_vector) 



            
        
            #Set return all grasps to true
            try:
                metric_scores = np.asarray(json_dict['metric'])
            except KeyError:
                metric_scores = np.ones(grasps.shape[0])
            return_all_grasps = False
        else:
            grasps[:, :3, 3] -= object_mean
            try:
                flex_qualities = np.asarray(json_dict[quality])
            except KeyError:
                flex_qualities = np.ones(grasps.shape)
            try:
                heuristic_qualities = np.asarray(
                    json_dict['quality_number_of_contacts'])
            except KeyError:
                heuristic_qualities = np.ones(flex_qualities.shape)
        # print(flex_qualities.shape, heuristic_qualities.shape)
        if len(grasps.shape) == 4:
            successful_mask = np.where(metric_scores >= 2.4)
            negative_mask = np.where(metric_scores < 2.4)
            positive_grasp_indexes = successful_mask[0]
            negative_grasp_indexes = negative_mask[0]
        else:
            successful_mask = np.logical_and(flex_qualities > 0.01,
                                            heuristic_qualities > 0.01) 
            positive_grasp_indexes = np.where(successful_mask)[0]
            negative_grasp_indexes = np.where(~successful_mask)[0]
        # positive_grasp_indexes = np.unique(positive_grasp_indexes)
        # print(len(positive_grasp_indexes))
        if len(grasps.shape) == 4:
            positive_grasps = grasps[positive_grasp_indexes, :, :, :]
            negative_grasps = grasps[negative_grasp_indexes, :, :, :]
            positive_qualities = metric_scores[positive_grasp_indexes]
            negative_qualities = metric_scores[negative_grasp_indexes]
        else:
            positive_grasps = grasps[positive_grasp_indexes, :, :]
            negative_grasps = grasps[negative_grasp_indexes, :, :]
            positive_qualities = heuristic_qualities[positive_grasp_indexes]
            negative_qualities = heuristic_qualities[negative_grasp_indexes]
        
        def cluster_grasps(grasps, qualities):
            cluster_indexes = np.asarray(
                utils.farthest_points(grasps, num_clusters,
                                      utils.distance_by_translation_grasp))
            output_grasps = []
            output_qualities = []

            for i in range(num_clusters):
                indexes = np.where(cluster_indexes == i)[0]
                if ratio_of_grasps_to_be_used < 1:
                    num_grasps_to_choose = max(
                        1,
                        int(ratio_of_grasps_to_be_used * float(len(indexes))))
                    if len(indexes) == 0:
                        raise NoPositiveGraspsException
                    indexes = np.random.choice(indexes,
                                               size=num_grasps_to_choose,
                                               replace=False)
                if(len(grasps.shape) == 4):
                    output_grasps.append(grasps[indexes, :, :, :])
                else:
                    output_grasps.append(grasps[indexes, :, :])
                output_qualities.append(qualities[indexes])

            output_grasps = np.asarray(output_grasps)
            output_qualities = np.asarray(output_qualities)

            return output_grasps, output_qualities
        if not return_all_grasps:
            positive_grasps, positive_qualities = cluster_grasps(
                positive_grasps, positive_qualities)
            negative_grasps, negative_qualities = cluster_grasps(
                negative_grasps, negative_qualities)
            num_positive_grasps = np.sum([p.shape[0] for p in positive_grasps])
            num_negative_grasps = np.sum([p.shape[0] for p in negative_grasps])
        else:            
            num_positive_grasps = positive_grasps.shape[0]
            num_negative_grasps = negative_grasps.shape[0]
        return positive_grasps, positive_qualities, negative_grasps, negative_qualities, object_model, os.path.join(
            root_folder, json_dict['object']), json_dict['object_scale']

    def sample_grasp_indexes(self, n, grasps, qualities):
        """
          Stratified sampling of the grasps.
        """
        nonzero_rows = [i for i in range(len(grasps)) if len(grasps[i]) > 0]
        num_clusters = len(nonzero_rows)
        replace = n > num_clusters
        if num_clusters == 0:
            raise NoPositiveGraspsException

        grasp_rows = np.random.choice(range(num_clusters),
                                      size=n,
                                      replace=replace).astype(np.int32)
        grasp_rows = [nonzero_rows[i] for i in grasp_rows]
        grasp_cols = []
        for grasp_row in grasp_rows:
            if len(grasps[grasp_rows]) == 0:
                raise ValueError('grasps cannot be empty')

            grasp_cols.append(np.random.randint(len(grasps[grasp_row])))

        grasp_cols = np.asarray(grasp_cols, dtype=np.int32)

        return np.vstack((grasp_rows, grasp_cols)).T

    def get_mean_std(self):
        """ Computes Mean and Standard Deviation from Training Data
        If mean/std file doesn't exist, will compute one
        :returns
        mean: N-dimensional mean
        std: N-dimensional standard deviation
        ninput_channels: N
        (here N=5)
        """

        mean_std_cache = os.path.join(self.opt.dataset_root_folder,
                                      'mean_std_cache.p')
        if not os.path.isfile(mean_std_cache):
            print('computing mean std from train data...')
            # doesn't run augmentation during m/std computation
            num_aug = self.opt.num_aug
            self.opt.num_aug = 1
            mean, std = np.array(0), np.array(0)
            for i, data in enumerate(self):
                if i % 500 == 0:
                    print('{} of {}'.format(i, self.size))
                features = data['edge_features']
                mean = mean + features.mean(axis=1)
                std = std + features.std(axis=1)
            mean = mean / (i + 1)
            std = std / (i + 1)
            transform_dict = {
                'mean': mean[:, np.newaxis],
                'std': std[:, np.newaxis],
                'ninput_channels': len(mean)
            }
            with open(mean_std_cache, 'wb') as f:
                pickle.dump(transform_dict, f)
            print('saved: ', mean_std_cache)
            self.opt.num_aug = num_aug
        # open mean / std from file
        with open(mean_std_cache, 'rb') as f:
            transform_dict = pickle.load(f)
            print('loaded mean / std from cache')
            self.mean = transform_dict['mean']
            self.std = transform_dict['std']
            self.ninput_channels = transform_dict['ninput_channels']

    def make_dataset(self):
        split_files = os.listdir(
            os.path.join(self.opt.dataset_root_folder,
                         self.opt.splits_folder_name))
        files = []
        for split_file in split_files:
            if split_file.find('.json') < 0:
                continue
            should_go_through = False
            if self.opt.allowed_categories == '':
                should_go_through = True
                if self.opt.blacklisted_categories != '':
                    if self.opt.blacklisted_categories.find(
                            split_file[:-5]) >= 0:
                        should_go_through = False
            else:
                if self.opt.allowed_categories.find(split_file[:-5]) >= 0:
                    should_go_through = True

            if should_go_through:
                files += [
                    os.path.join(self.opt.dataset_root_folder,
                                 self.opt.grasps_folder_name, f)
                    for f in json.load(
                        open(
                            os.path.join(self.opt.dataset_root_folder,
                                         self.opt.splits_folder_name,
                                         split_file)))[self.opt.dataset_split]
                ]
        return files


def collate_fn(batch):
    """Creates mini-batch tensors
    We should build custom collate_fn rather than using default collate_fn
    """
    batch = list(filter(lambda x: x is not None, batch))  #
    meta = {}
    keys = batch[0].keys()
    for key in keys:
        meta.update({key: np.concatenate([d[key] for d in batch])})
    return meta
