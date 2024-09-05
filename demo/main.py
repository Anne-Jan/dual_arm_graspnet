from __future__ import print_function

import numpy as np
import argparse
import grasp_estimator
import sys
import os
import glob
import mayavi.mlab as mlab
from utils.visualization_utils import *
import mayavi.mlab as mlab
from utils import utils
from data import DataLoader
import trimesh.transformations as tra


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
        '--choose_fn',
        choices={
            "all", "better_than_threshold", "better_than_threshold_in_sequence"
        },
        default='better_than_threshold',
        help=
        "If all, no grasps are removed. If better than threshold, only the last refined grasps are considered while better_than_threshold_in_sequence consideres all refined grasps"
    )

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


def main(args):
    parser = make_parser()
    args = parser.parse_args()
    grasp_sampler_args = utils.read_checkpoint_args(args.grasp_sampler_folder)
    grasp_sampler_args.is_train = False

    grasp_evaluator_args = utils.read_checkpoint_args(
        args.grasp_evaluator_folder)
    grasp_evaluator_args.continue_train = True
    estimator = grasp_estimator.GraspEstimator(grasp_sampler_args,
                                               grasp_evaluator_args, args)
    if args.train_data:
        grasp_sampler_args.dataset_root_folder = args.dataset_root_folder
        grasp_sampler_args.num_grasps_per_object = 1
        grasp_sampler_args.num_objects_per_batch = 1
        dataset = DataLoader(grasp_sampler_args)
        
        for i, data in enumerate(dataset):
            #works, uncomment to use the vae and evaluator
            generated_grasps, generated_scores = estimator.generate_and_refine_grasps(
                data["pc"].squeeze())
            # generated_grasps = None
            # print(np.array(generated_grasps).shape)
            if generated_grasps is None:
                S = np.diag([0.2, 0.2, 0.2, 1])
                pc, _ = estimator.prepare_pc(data["pc"].squeeze())
                generated_grasps, _, _ = estimator.generate_grasps(pc)
                #reshape to n by 14
                # generated_grasps = generated_grasps.reshape(-1, 14)
                # generated_grasps = utils.transform_control_points(
                #     generated_grasps[0], batch_size = len(generated_grasps[0]),mode='qt', device='cuda').cpu()
                grasps = []
                for grasp in generated_grasps:
                    # print("grasp", grasp.shape)
                    euler_angles, translations = utils.convert_qt_to_rt(grasp.cpu())
                    selection = np.ones((len(euler_angles), 1))
                    refine_indexes, sample_indexes = np.where(selection)
                    euler_angles1 = euler_angles[:, :3]
                    euler_angles2 = euler_angles[:, 3:]
                    translations1 = translations[:, :3]
                    translations2 = translations[:, 3:]
                    for refine_index, sample_index in zip(refine_indexes, sample_indexes):
                    # print("refine_index", refine_index, "sample_index", sample_index)
                        rt1 = tra.euler_matrix(*euler_angles1[refine_index, :])
                        rt1[:3, 3] = translations1[refine_index, :]
                        rt2 = tra.euler_matrix(*euler_angles2[refine_index, :])
                        rt2[:3, 3] = translations2[refine_index, :]
                        #Combine them into nx2x4x4 instead of nx4x4
                        # rt1 = np.expand_dims(rt1, 0)
                        # rt2 = np.expand_dims(rt2, 0)
                        # print("rt1", rt1.shape, rt2.shape)
                        # print("concatenated", np.concatenate((rt1, rt2), 0).shape)
                        #Simply append them both to the list
                        # rt1 = S.dot(rt1)
                        # rt2 = S.dot(rt2)
                        grasps.append(rt1)
                        grasps.append(rt2)
                        generated_grasps = grasps
            # else:
                # print("scaling")
                # s = np.diag([0.2, 0.2, 0.2, 1])
                # for i in range(len(generated_grasps)):
                #     generated_grasps[i][0] = s.dot(generated_grasps[i][0])
                #     generated_grasps[i][1] = s.dot(generated_grasps[i][1])
        

            print(np.array(generated_grasps).shape)
            mlab.figure(bgcolor=(1, 1, 1))
            # draw_scene(data["pc"][0],
            #            grasps=generated_grasps,
            #            grasp_scores=generated_scores)
            draw_scene(data["pc"][0],
                       grasps=generated_grasps)
            print('close the window to continue to next object . . .')
            mlab.show()
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
            print(np.array(generated_grasps).shape)
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
