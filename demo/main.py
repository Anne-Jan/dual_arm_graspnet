from __future__ import print_function

import numpy as np
import argparse
import grasp_estimator
import torch
import sys
import os
import glob
import mayavi.mlab as mlab
from utils.visualization_utils import *
import mayavi.mlab as mlab
from utils import utils
from data import DataLoader
import trimesh.transformations as tra
from scipy.spatial.transform import Rotation as R


def make_parser():
    parser = argparse.ArgumentParser(
        description='6-DoF GraspNet Demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--grasp_sampler_folder',
                        type=str,
                        default='checkpoints/gan_pretrained/')
    parser.add_argument('--grasp_evaluator_folder',
                        type=str,
                        default='checkpoints/evaluator_pretrained/')
    parser.add_argument('--refinement_method',
                        choices={"gradient", "sampling"},
                        default='sampling')
    parser.add_argument('--refine_steps', type=int, default=25)

    parser.add_argument('--npy_folder', type=str, default='demo/data/')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.8,
        help=
        "When choose_fn is something else than all, all grasps with a score given by the evaluator notwork less than the threshold are removed"
    )
    parser.add_argument(
        '--object_scale',
        type=float,
        default=0.2,
        help=
        "Sets the downscaled size of the object. The object is scaled to this size before being fed to the network"
    )
    parser.add_argument(
        '--choose_fn',
        choices={
            "all", "better_than_threshold", "better_than_threshold_in_sequence"
        },
        default='better_than_threshold',
        help=
        "If all, no grasps are removed. If better than threshold, only the last refined grasps are considered while better_than_threshold_in_sequence consideres all refined grasps"
    )
    parser.add_argument('--num_objects_to_show', type=int, default=5)
    parser.add_argument('--target_pc_size', type=int, default=1024)
    parser.add_argument('--num_grasp_samples', type=int, default=200)
    parser.add_argument(
        '--generate_dense_grasps',
        action='store_true',
        help=
        "If enabled, it will create a [num_grasp_samples x num_grasp_samples] dense grid of latent space values and generate grasps from these."
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=30,
        help=
        "Set the batch size of the number of grasps we want to process and can fit into the GPU memory at each forward pass. The batch_size can be increased for a GPU with more memory."
    )
    parser.add_argument('--train_data', action='store_true')
    opts, _ = parser.parse_known_args()
    if opts.train_data:
        parser.add_argument('--dataset_root_folder',
                            required=True,
                            type=str,
                            help='path to root directory of the dataset.')
    parser.add_argument("--dual_grasp",
                        action="store_true",
                        help="If enabled, the model will predict two grasps.")
    return parser


def get_color_for_pc(pc, K, color_image):
    proj = pc.dot(K.T)
    proj[:, 0] /= proj[:, 2]
    proj[:, 1] /= proj[:, 2]

    pc_colors = np.zeros((pc.shape[0], 3), dtype=np.uint8)
    for i, p in enumerate(proj):
        x = int(p[0])
        y = int(p[1])
        pc_colors[i, :] = color_image[y, x, :]

    return pc_colors


def backproject(depth_cv,
                intrinsic_matrix,
                return_finite_depth=True,
                return_selection=False):

    depth = depth_cv.astype(np.float32, copy=True)

    # get intrinsic matrix
    K = intrinsic_matrix
    Kinv = np.linalg.inv(K)

    # compute the 3D points
    width = depth.shape[1]
    height = depth.shape[0]

    # construct the 2D points matrix
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width * height, 3)

    # backprojection
    R = np.dot(Kinv, x2d.transpose())

    # compute the 3D points
    X = np.multiply(np.tile(depth.reshape(1, width * height), (3, 1)), R)
    X = np.array(X).transpose()
    if return_finite_depth:
        selection = np.isfinite(X[:, 0])
        X = X[selection, :]

    if return_selection:
        return X, selection

    return X

# Helper function to compute the volume of a bounding box
def bounding_box_volume(bbox):
    x_len, y_len, z_len = bbox
    return x_len * y_len * z_len# Function to compute the intersection volume of two 3D bounding boxes

def intersection_volume(bbox1, bbox2, pos1, pos2):
    # Calculate the overlap in each dimension
    overlap_x = max(0, min(pos1[0] + bbox1[0] / 2, pos2[0] + bbox2[0] / 2) - max(pos1[0] - bbox1[0] / 2, pos2[0] - bbox2[0] / 2))
    overlap_y = max(0, min(pos1[1] + bbox1[1] / 2, pos2[1] + bbox2[1] / 2) - max(pos1[1] - bbox1[1] / 2, pos2[1] - bbox2[1] / 2))
    overlap_z = max(0, min(pos1[2] + bbox1[2] / 2, pos2[2] + bbox2[2] / 2) - max(pos1[2] - bbox1[2] / 2, pos2[2] - bbox2[2] / 2))    # Intersection volume
    return overlap_x * overlap_y * overlap_z# Function to calculate IoU for 6D grasps


def grasp_iou(pose1, pose2, bbox1 = [0.03, 0.03, 0.03], bbox2 = [0.03, 0.03, 0.03], threshold = 0.25):
    # Extract positions and orientations from poses

    pos1, pos2 = pose1[:3, 3], pose2[:3, 3]  # (x, y, z)    # Compute the intersection and union volumes
    intersection_vol = intersection_volume(bbox1, bbox2, pos1, pos2)
    rot1, rot2 = pose1[:3, :3], pose2[:3, :3]  # Compute the intersection and union volumes
    union_vol = bounding_box_volume(bbox1) + bounding_box_volume(bbox2) - intersection_vol    # Calculate IoU
    iou = intersection_vol / union_vol    # Return 1 if IoU is above or equal to threshold, else return 0
    # print (iou)
    if iou >= threshold:
        # print("IoU", iou)
        #Check if the rotation is within 40 degrees, if not return 0
        if np.any(np.abs(rotation_angle_difference(rot1, rot2)) > 40):
            return 0
        else:
            return 1
    return 0

def rotation_angle_difference(R1, R2):
    # R1 and R2 are 3x3 rotation matrices
    R_diff = np.dot(R1.T, R2)  # Difference in rotation
    r = R.from_matrix(R_diff)
    euler_angles = r.as_euler('zyx', degrees=True)
    # print("euler_angles", euler_angles)
    # axis_angle = r.as_rotvec()
    # angle_deg = np.degrees(np.linalg.norm(axis_angle))
    return euler_angles# Function to calculate the angle between two rotation matrices

def main(args):
    parser = make_parser()
    args = parser.parse_args()
    grasp_sampler_args = utils.read_checkpoint_args(args.grasp_sampler_folder)
    grasp_sampler_args.is_train = False
    num_objects_to_show = args.num_objects_to_show
    grasp_evaluator_args = utils.read_checkpoint_args(
        args.grasp_evaluator_folder)
    grasp_evaluator_args.continue_train = True
    estimator = grasp_estimator.GraspEstimator(grasp_sampler_args,
                                               grasp_evaluator_args, args)
    all_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    object_succesfull_for_08 = None
    if args.train_data:
        for jaccard_threshold in all_thresholds:
            grasp_sampler_args.dataset_root_folder = args.dataset_root_folder
            grasp_sampler_args.num_grasps_per_object = 1
            grasp_sampler_args.num_objects_per_batch = 1
            dataset = DataLoader(grasp_sampler_args)
            # jaccard_threshold = 0.8
            objects_shown = 0
            num_objects_with_good_grasps = 0
            total_grasp_pairs_found_for_threshold = 0
            print("Starting to process objects for threshold", jaccard_threshold)
            for i, data in enumerate(dataset):
                # print("Processing object: ", i+1, 'which is', objects_shown, 'of total to process', num_objects_to_show)
                # mlab.figure(bgcolor=(1, 1, 1))
                # #draw a box of size 0.02, 0.02, 0.02
                # mlab.points3d(0, 0, 0, scale_factor=0.03, color=(1, 0, 0), mode = 'cube')
                # grasp = data["all_target_cps"][0][0]
                # mlab.points3d(grasp[:, 0], grasp[:, 1], grasp[:, 2], scale_factor=0.01, color=(0, 1, 0))
                # mlab.show()
                # print(xd)    
                # already_processed = ['shapenet_models/da2_dataset_small/meshes/2bf0144b097a81c2639c1eedc5ef16cc.stl','shapenet_models/da2_dataset_small/meshes/3959f60f4e4356b35817e30de1dabac4.stl','shapenet_models/da2_dataset_small/meshes/fd9b63c23342e57045b799df37b9f05.stl','shapenet_models/da2_dataset_small/meshes/4698973d47c780654f48c7d5931203ac.stl', 'shapenet_models/da2_dataset_small/meshes/c17873ebacc5476e1ee2b36e795b09b7.stl', 'shapenet_models/da2_dataset_small/meshes/ca3e6e011ed7ecb32dc86554a873c4fe.stl','shapenet_models/da2_dataset_small/meshes/b2cfba8ee63abd118fac6a8030e15671.stl']
                # if data['cad_path'] in already_processed:
                #     continue
                #works, uncomment to use the vae and evaluator
                # if object_succesfull_for_08 != None:
                #     print("Setting data to object succesfull for 08")
                #     data = object_succesfull_for_08
                good_grasps = 0
                grasps_to_save=[]
                # print("Generating grasps")
                generated_grasps, generated_scores = estimator.generate_and_refine_grasps(
                    data["pc"].squeeze(), data['all_target_cps'])
                # if generated_grasps is None:
                #     print("No grasps were generated")
                #     continue
                #Visualize the grasps before comparison to ground truth
                # mlab.figure(bgcolor=(1, 1, 1))
                # draw_scene(data["pc"][0],
                #            grasps=generated_grasps,)
                        #    target_cps = data['all_target_cps'][:500].reshape(-1,6,3),)
                        #    grasp_scores=generated_scores)
               
                # print(data["all_grasp_rt"].shape)
                # print("non reshaped", data["all_grasp_rt"][0])
                grasp_rt = data["all_grasp_rt"].reshape(-1, 4, 4)

                #Compare the generated grasps with the ground truth grasps
                print("Found", len(generated_grasps), "grasps. Calculating jaccard index")
                #take the grasp pairs with the highest score
                # index_of_score = np.argmax(generated_scores)
                # grasp1 = generated_grasps[2 * index_of_score]
                # grasp2 = generated_grasps[(2 * index_of_score) + 1]
                # generated_grasps = [grasp1, grasp2]
                # print(len(generated_grasps))
                # mlab.figure(bgcolor=(1, 1, 1))
                # draw_scene(data["pc"][0],
                #               grasps=generated_grasps,)
                #             #   target_cps=torch.FloatTensor(grasp_pcs))
                # print('close the window to continue to next object . . .')
                # mlab.show()
                for i in range(len(generated_grasps) -1):
                    if i %2 != 0 and i != 0:
                        continue
                    for j in range(len(grasp_rt)-1):
                        if j %2 != 0 and j != 0:
                            continue
                        # print("jaccard_index_with_threshold_and_rotation", jaccard_index_with_threshold_and_rotation(generated_grasps[i], grasp_rt[j]))
                        # print(i, j)
                        if (grasp_iou(generated_grasps[i], grasp_rt[j], threshold= jaccard_threshold) == 1 and (grasp_iou(generated_grasps[i+1], grasp_rt[j+1], threshold= jaccard_threshold) == 1)) or (grasp_iou(generated_grasps[i], grasp_rt[j+1], threshold=jaccard_threshold) == 1 and(grasp_iou(generated_grasps[i+1], grasp_rt[j], threshold=jaccard_threshold) == 1)):
                            good_grasps += 1
                            grasps_to_save.append(generated_grasps[i])
                            grasps_to_save.append(generated_grasps[i+1])
                            # print("grasp found")
                            break
                        # else:
                        #     rejected_grasps.append(generated_grasps[i])
                        #     rejected_grasps.append(generated_grasps[i+1])
                            # break
                #Visualize some rejected grasps
                # mlab.figure(bgcolor=(1, 1, 1))
                # draw_scene_dual(data["pc"][0],
                #            grasps=rejected_grasps[:20],)
                # mlab.show()
                generated_grasps = grasps_to_save
                if good_grasps == 0:
                    print("No good grasps were found")
                    #only for visualisation, should be removed if calculating jaccard index
                    # continue
                else:
                    print("Found", good_grasps, "good grasps")
                    # print("File:", data['cad_path'])
                    mlab.figure(bgcolor=(1, 1, 1))
                    draw_scene(data["pc"][0],
                                  grasps=generated_grasps,)
                                #   target_cps=torch.FloatTensor(grasp_pcs))
                    print('close the window to continue to next object . . .')
                    mlab.show()
                    total_grasp_pairs_found_for_threshold += good_grasps
                    num_objects_with_good_grasps += 1
                    # continue
                
                #double the grasp scores pairwise. So that for example index 0 and 1 of the new scores have the same score as index 0 of the old scores
                # generated_scores = np.concatenate((generated_scores, generated_scores), axis = 0)
                # mlab.figure(bgcolor=(1, 1, 1))
                # draw_scene_dual(data["pc"][0],
                #            grasps=generated_grasps,)
                # mlab.show()
                #Found a good object, save it and continue to the next threshold
                # object_succesfull_for_08 = data
                # break
                        #    target_cps = data['all_target_cps'][:500].reshape(-1,6,3),)
                        #    grasp_scores=generated_scores)
                # # draw_scene(data["pc"][0],
                # #               grasps=generated_grasps,
                # #               target_cps=torch.FloatTensor(grasp_pcs))
                # print('close the window to continue to next object . . .')
                # mlab.show()
                objects_shown += 1
                if objects_shown == num_objects_to_show:
                    print("Found", num_objects_with_good_grasps, "objects with good grasps for threshold", jaccard_threshold)
                    print("Average number of good grasps found for this threshold", total_grasp_pairs_found_for_threshold/num_objects_with_good_grasps)
                    break
    else:
        for npy_file in glob.glob(os.path.join(args.npy_folder, '*.npy')):
            # Depending on your numpy version you may need to change allow_pickle
            # from True to False.

            data = np.load(npy_file, allow_pickle=True,
                           encoding="latin1").item()

            depth = data['depth']
            image = data['image']
            K = data['intrinsics_matrix']
            # Removing points that are farther than 1 meter or missing depth
            # values.
            #depth[depth == 0 or depth > 1] = np.nan

            np.nan_to_num(depth, copy=False)
            mask = np.where(np.logical_or(depth == 0, depth > 1))
            depth[mask] = np.nan
            pc, selection = backproject(depth,
                                        K,
                                        return_finite_depth=True,
                                        return_selection=True)
            pc_colors = image.copy()
            pc_colors = np.reshape(pc_colors, [-1, 3])
            pc_colors = pc_colors[selection, :]

            # Smoothed pc comes from averaging the depth for 10 frames and removing
            # the pixels with jittery depth between those 10 frames.
            object_pc = data['smoothed_object_pc']
            generated_grasps, generated_scores = estimator.generate_and_refine_grasps(
                object_pc)
            # Only show the first grasp
            # generated_grasps = generated_grasps[0:2]
            # generated_scores = generated_scores[0:2]
            print(np.array(generated_grasps).shape, np.array(generated_scores).shape)
            mlab.figure(bgcolor=(1, 1, 1))
            draw_scene(
                pc,
                pc_color=pc_colors,
                grasps=generated_grasps,
                grasp_scores=generated_scores,
            )
            print('close the window to continue to next object . . .')
            mlab.show()
            #Only show the first object
            break


if __name__ == '__main__':
    main(sys.argv[1:])
