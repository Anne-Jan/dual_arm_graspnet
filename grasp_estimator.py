from __future__ import print_function

from models import create_model
import numpy as np
import torch
import time
import trimesh
import trimesh.transformations as tra
#import surface_normal
import copy
import os
from utils import utils

class GraspEstimator:
    """
      Includes the code used for running the inference.
    """
    def __init__(self, grasp_sampler_opt, grasp_evaluator_opt, opt):
        self.grasp_sampler_opt = grasp_sampler_opt
        self.grasp_evaluator_opt = grasp_evaluator_opt
        try:
            self.grasp_evaluator_opt.dual_grasp = grasp_sampler_opt['dual_grasp']
            self.grasp_sampler_opt.dual_grasp = grasp_sampler_opt['dual_grasp']
        except:
            self.grasp_evaluator_opt.dual_grasp = False
            self.grasp_sampler_opt.dual_grasp = False
        self.opt = opt
        self.target_pc_size = opt.target_pc_size
        self.num_refine_steps = opt.refine_steps
        self.refine_method = opt.refinement_method
        self.threshold = opt.threshold
        self.batch_size = opt.batch_size
        self.generate_dense_grasps = opt.generate_dense_grasps
        if self.generate_dense_grasps:
            self.num_grasps_per_dim = opt.num_grasp_samples
            self.num_grasp_samples = opt.num_grasp_samples * opt.num_grasp_samples
        else:
            self.num_grasp_samples = opt.num_grasp_samples
        self.choose_fn = opt.choose_fn
        self.choose_fns = {
            "all":
            None,
            "better_than_threshold":
            utils.choose_grasps_better_than_threshold,
            "better_than_threshold_in_sequence":
            utils.choose_grasps_better_than_threshold_in_sequence,
        }
        self.device = torch.device("cuda:0")
        self.grasp_evaluator = create_model(grasp_evaluator_opt)
        self.grasp_sampler = create_model(grasp_sampler_opt)

    def keep_inliers(self, grasps, confidences, z, pc, inlier_indices_list):
        # print("before", grasps[0].shape)
        for i, inlier_indices in enumerate(inlier_indices_list):
            
            grasps[i] = grasps[i][inlier_indices]
            confidences[i] = confidences[i][inlier_indices[0]]
            # print(len(confidences))
            z[i] = z[i][inlier_indices]
            pc[i] = pc[i][inlier_indices]
        # print("after", grasps[0].shape)

    def generate_and_refine_grasps(
        self,
        pc,
    ):
        pc_list, pc_mean = self.prepare_pc(pc)
        grasps_list, confidence_list, z_list = self.generate_grasps(pc_list)
        print("grasps per pc",grasps_list[0].shape)
        inlier_indices = utils.get_inlier_grasp_indices(grasps_list,
                                                        torch.zeros(1, 3).to(
                                                            self.device),
                                                        threshold=1.0,
                                                        device=self.device)
        # print(np.array(grasps_list).shape, np.array(confidence_list).shape, np.array(z_list).shape, np.array(pc_list).shape, np.array(inlier_indices).shape)
        self.keep_inliers(grasps_list, confidence_list, z_list, pc_list,
                          inlier_indices)
        improved_eulers, improved_ts, improved_success = [], [], []
        
        for pc, grasps in zip(pc_list, grasps_list):
            out = self.refine_grasps(pc, grasps, self.refine_method,
                                     self.num_refine_steps, og_pc=pc[0])
            if len(out[0]) == 0:
                print("No grasps found")
                continue
            improved_eulers.append(out[0])
            improved_ts.append(out[1])
            improved_success.append(out[2])
        
        # print(np.array(improved_eulers).shape)
        # print(np.array(improved_ts).shape)
        # print(np.array(improved_success).shape)
        # print("improved eulers",np.array(improved_eulers).shape, "improved ts", np.array(improved_ts).shape, "improved success", np.array(improved_success).shape)
        improved_eulers = np.hstack(improved_eulers)
        improved_ts = np.hstack(improved_ts)
        improved_success = np.hstack(improved_success)


        if self.choose_fn is "all":
            selection_mask = np.ones(improved_success.shape, dtype=np.float32)
        else:
            selection_mask = self.choose_fns[self.choose_fn](improved_eulers,
                                                             improved_ts,
                                                             improved_success,
                                                             self.threshold)
        grasps = utils.rot_and_trans_to_grasps(improved_eulers, improved_ts,
                                               selection_mask)
        # utils.denormalize_grasps(grasps, pc_mean)
        refine_indexes, sample_indexes = np.where(selection_mask)
        success_prob = improved_success[refine_indexes,
                                        sample_indexes].tolist()
        # grasps, success_prob = self.final_selection(grasps, success_prob, pc[0])
        return grasps, success_prob

    def prepare_pc(self, pc):
        # print(self.batch_size)
        if pc.shape[0] > self.target_pc_size:
            pc = utils.regularize_pc_point_count(pc, self.target_pc_size)
        pc_mean = np.mean(pc, 0)
        pc -= np.expand_dims(pc_mean, 0)
        pc = np.tile(pc, (self.num_grasp_samples, 1, 1))
        pc = torch.from_numpy(pc).float().to(self.device)
        pcs = []
        pcs = utils.partition_array_into_subarrays(pc, self.batch_size)
        # for i in range(self.batch_size):
        #     pcs.append(pc)
        return pcs, pc_mean

    def generate_grasps(self, pcs):
        all_grasps = []
        all_confidence = []
        all_z = []
        if self.generate_dense_grasps:
            latent_samples = self.grasp_sampler.net.module.generate_dense_latents(
                self.num_grasps_per_dim)
            latent_samples = utils.partition_array_into_subarrays(
                latent_samples, self.batch_size)
            for latent_sample, pc in zip(latent_samples, pcs):
                grasps, confidence, z = self.grasp_sampler.generate_grasps(
                    pc, latent_sample)
                all_grasps.append(grasps)
                all_confidence.append(confidence)
                all_z.append(z)
        else:
            for pc in pcs:
                grasps, confidence, z = self.grasp_sampler.generate_grasps(pc)
                # print(grasps.shape)
                all_grasps.append(grasps)
                all_confidence.append(confidence)
                all_z.append(z)
                # print("grasps",all_grasps[0].shape)
        return all_grasps, all_confidence, all_z

    def refine_grasps(self, pc, grasps, refine_method, num_refine_steps=10, og_pc=None):
        grasp_eulers, grasp_translations = utils.convert_qt_to_rt(grasps)
        if refine_method == "gradient":
            improve_fun = self.improve_grasps_gradient_based
            if len(grasp_eulers[0]) == 6:
                # grasp_eulers1 = grasp_eulers[:, :3]
                # grasp_translations1 = grasp_translations[:, :3]
                # grasp_eulers2 = grasp_eulers[:, 3:]
                # grasp_translations2 = grasp_translations[:, 3:]
                grasp_eulers1, grasp_eulers2 = torch.split(grasp_eulers, 3, dim = 1)
                grasp_translations1, grasp_translations2 = torch.split(grasp_translations, 3, dim = 1)
                grasp_eulers1 = torch.autograd.Variable(grasp_eulers1.to(
                    self.device),
                                                        requires_grad=True)
                grasp_translations1 = torch.autograd.Variable(
                    grasp_translations1.to(self.device),
                    requires_grad=True)
                grasp_eulers2 = torch.autograd.Variable(grasp_eulers2.to(
                    self.device),
                                                        requires_grad=True)
                grasp_translations2 = torch.autograd.Variable(
                    grasp_translations2.to(self.device),
                    requires_grad=True)
                grasp_eulers = torch.cat((grasp_eulers1, grasp_eulers2), -1)
                grasp_translations = torch.cat(
                    (grasp_translations1, grasp_translations2), -1)
            else:
                grasp_eulers = torch.autograd.Variable(grasp_eulers.to(
                    self.device),
                                                    requires_grad=True)
                grasp_translations = torch.autograd.Variable(grasp_translations.to(
                    self.device),
                                                        requires_grad=True)

        else:
            improve_fun = self.improve_grasps_sampling_based
            # improve_fun = self.improve_grasps_sampling_based_new

        if refine_method == "gradient":
            ###Old Code
            improved_success = []
            improved_eulers = []
            improved_ts = []
            improved_eulers.append(grasp_eulers.cpu().data.numpy())
            improved_ts.append(grasp_translations.cpu().data.numpy())
            last_success = None
            for i in range(num_refine_steps):
                success_prob, last_success = improve_fun(pc, grasp_eulers1, grasp_eulers2,
                                                        grasp_translations1, grasp_translations2,
                                                        last_success, og_pc)
                improved_success.append(success_prob.cpu().data.numpy())
                improved_eulers.append(grasp_eulers.cpu().data.numpy())
                improved_ts.append(grasp_translations.cpu().data.numpy())

            # we need to run the success on the final improved grasps
            grasp_pcs = utils.control_points_from_rot_and_trans(
                grasp_eulers, grasp_translations, self.device,pc = None)
            improved_success.append(
                self.grasp_evaluator.evaluate_grasps(
                    pc, grasp_pcs).squeeze().cpu().data.numpy())
            ###Old Code
        elif refine_method == "sampling" and improve_fun.__name__ == "improve_grasps_sampling_based":
            improved_success = []
            improved_eulers = []
            improved_ts = []
            improved_eulers.append(grasp_eulers.cpu().data.numpy())
            improved_ts.append(grasp_translations.cpu().data.numpy())
            last_success = None
            for i in range(num_refine_steps):
                success_prob, last_success = improve_fun(pc, grasp_eulers,
                                                         grasp_translations,
                                                         last_success, og_pc)
                improved_success.append(success_prob.cpu().data.numpy())
                improved_eulers.append(grasp_eulers.cpu().data.numpy())
                improved_ts.append(grasp_translations.cpu().data.numpy())

            # we need to run the success on the final improved grasps
            grasp_pcs = utils.control_points_from_rot_and_trans(
                grasp_eulers, grasp_translations, self.device,pc = None)
            improved_success.append(
                self.grasp_evaluator.evaluate_grasps(
                    pc, grasp_pcs).squeeze().cpu().data.numpy())
        else:
            ###New Cod # 
            improved_success = []
            improved_eulers = []
            improved_ts = []
            for i in range(num_refine_steps):
                eulers, ts, success = improve_fun(pc, grasp_eulers, grasp_translations, None, og_pc, self.threshold)
                #check if the eulers and translations are empty
                if len(eulers) == 0:
                    # print("No grasps found")
                    continue      
                else:      
                    improved_eulers.append(eulers)
                    improved_ts.append(ts)
                    improved_success.append(success)
        ###New Code
        return np.asarray(improved_eulers), np.asarray(
            improved_ts), np.asarray(improved_success)

    def improve_grasps_gradient_based(
        self, pcs, grasp_eulers1, grasp_eulers2, grasp_trans1, grasp_trans2, last_success, og_pc = None
    ):  #euler_angles, translation, eval_and_improve, metadata):
        grasp_eulers = torch.cat((grasp_eulers1, grasp_eulers2), -1)
        grasp_trans= torch.cat(
            (grasp_trans1, grasp_trans2), -1)
        grasp_pcs = utils.control_points_from_rot_and_trans(
            grasp_eulers, grasp_trans, self.device, pc=og_pc)
        success = self.grasp_evaluator.evaluate_grasps(pcs, grasp_pcs)
        success.squeeze().backward(
            torch.ones(success.shape[0]).to(self.device))
        
        # print(torch.round(success))
        #split the grasp_eulers and grasp_translations into two parts
        # if len(grasp_eulers[1]) == 6:
        #     grasp_eulers1, grasp_eulers2 = torch.split(grasp_eulers, 3, dim = 1)
        #     grasp_trans1, grasp_trans2 = torch.split(grasp_trans, 3, dim = 1)
        if len(grasp_eulers[1]) == 6:
            # grasp_eulers1.retain_grad()
            # grasp_eulers2.retain_grad()
            # grasp_trans1.retain_grad()
            # grasp_trans2.retain_grad()
            delta_t1 = grasp_trans1.grad
            delta_t2 = grasp_trans2.grad
            norm_t1 = torch.norm(delta_t1, p=2, dim=-1).to(self.device)
            norm_t2 = torch.norm(delta_t2, p=2, dim=-1).to(self.device)
            alpha1 = torch.min(0.01 / norm_t1, torch.tensor(1.0).to(self.device))
            alpha2 = torch.min(0.01 / norm_t2, torch.tensor(1.0).to(self.device))
            grasp_trans1.data += grasp_trans1.grad * alpha1[:, None]
            grasp_trans2.data += grasp_trans2.grad * alpha2[:, None]
            grasp_eulers1.data += grasp_eulers1.grad * alpha1[:, None]
            grasp_eulers2.data += grasp_eulers2.grad * alpha2[:, None]
            # grasp_eulers1.grad.zero_()
            # grasp_eulers2.grad.zero_()
            # grasp_trans1.grad.zero_()
            # grasp_trans2.grad.zero_()
        else:
            delta_t = grasp_trans.grad
            print("delta_t", delta_t)
            norm_t = torch.norm(delta_t, p=2, dim=-1).to(self.device)
            print("norm_t", norm_t, "delta_t", delta_t)
            # Adjust the alpha so that it won't update more than 1 cm. Gradient is only valid
            # in small neighborhood.
            alpha = torch.min(0.01 / norm_t, torch.tensor(1.0).to(self.device))
            grasp_trans.data += grasp_trans.grad * alpha[:, None]
            temp = grasp_eulers.clone()
            grasp_eulers.data += grasp_eulers.grad * alpha[:, None]
        return success.squeeze(), None

    def improve_grasps_sampling_based(self,
                                      pcs,
                                      grasp_eulers,
                                      grasp_trans,
                                      last_success=None, og_pc = None):
        with torch.no_grad():
            if last_success is None:
                grasp_pcs = utils.control_points_from_rot_and_trans(
                    grasp_eulers, grasp_trans, self.device, pc = None)
                last_success = self.grasp_evaluator.evaluate_grasps(
                    pcs, grasp_pcs)
                # print("last success", last_success)
            delta_t = 2 * (torch.rand(grasp_trans.shape).to(self.device) - 0.5)
            delta_t *= 0.02
            delta_euler_angles = (
                torch.rand(grasp_eulers.shape).to(self.device) - 0.5) * 2
            perturbed_translation = grasp_trans + delta_t
            perturbed_euler_angles = grasp_eulers + delta_euler_angles
            grasp_pcs = utils.control_points_from_rot_and_trans(
                perturbed_euler_angles, perturbed_translation, self.device, pc = None)

            perturbed_success = self.grasp_evaluator.evaluate_grasps(
                pcs, grasp_pcs)
            # print("perturbed success", perturbed_success)
            ratio = perturbed_success / torch.max(
                last_success,
                torch.tensor(0.0001).to(self.device))
            mask = torch.rand(ratio.shape).to(self.device) <= ratio

            next_success = last_success
            ind = torch.where(mask)[0]
            next_success[ind] = perturbed_success[ind]
            grasp_trans[ind].data = perturbed_translation.data[ind]
            grasp_eulers[ind].data = perturbed_euler_angles.data[ind]
            return last_success.squeeze(), next_success
        

    def improve_grasps_sampling_based_new(self,
                                      pcs,
                                      grasp_eulers,
                                      grasp_trans,
                                      last_success=None, og_pc = None, threshold = 0.6):
        succesfull_eulers = []
        succesfull_trans = []
        succesfull_success = []
        with torch.no_grad():
            if last_success is None:
                grasp_pcs = utils.control_points_from_rot_and_trans(
                    grasp_eulers, grasp_trans, self.device, pc = None)
                last_success = self.grasp_evaluator.evaluate_grasps(
                    pcs, grasp_pcs)
                # print("last success", last_success)
        # 
        # print("last success", last_success)
        #print the max of the last success
        # print("max of last success", torch.max(last_success))
        for grasp_euler, grasp_tran, score in zip(grasp_eulers, grasp_trans, last_success):
            if score > threshold:
                grasp_euler = grasp_euler.cpu().data.numpy()
                grasp_tran = grasp_tran.cpu().data.numpy()
                score = score.cpu().data.numpy()
                score = score[0]
                # print(grasp_euler.shape, grasp_tran.shape, score)
                # print("Found a good grasp")
                if succesfull_eulers == []:
                    succesfull_eulers = grasp_euler
                    succesfull_trans = grasp_tran
                    succesfull_success.append(score)
                else:
                    #Add a new grasp to the list so that the become n by 6
                    succesfull_eulers = np.vstack((succesfull_eulers, grasp_euler))
                    succesfull_trans = np.vstack((succesfull_trans, grasp_tran))
                    succesfull_success.append(score)
        
        if len(succesfull_success) == 1:
            #Reshape the euleurs and translations to be 1 by 6
            # print(succesfull_eulers.shape, succesfull_trans.shape, np.array(succesfull_success).shape)
            succesfull_eulers = np.reshape(succesfull_eulers, (1, 6))
            succesfull_trans = np.reshape(succesfull_trans, (1, 6))
            # print(succesfull_eulers.shape, succesfull_trans.shape, np.array(succesfull_success).shape)
        return succesfull_eulers, succesfull_trans, succesfull_success

                
               
    def final_selection(self, grasps, succes_prob, og_pc):
        control_points = utils.control_points_from_grasps(grasps, 'cp', pc = None)
        print("ammount of grasps before", len(grasps), np.array(grasps).shape, control_points.shape)
        #For each grasp pair, check if the control points enclose the point cloud
        #If they do, then add the grasp to the list of succesfull grasps
        succesfull_grasps = []
        succesfull_prob = []
        grippper_left_widths = []
        gripper_right_widths = []
        threshold_distance = 0.01
        collision_threshold = 0.005
        unsuccesfull_grasps = []
        numpy_og_pc = og_pc.cpu().numpy()
        for idx in range(len(control_points) -1):
            if idx % 2 != 0:
                continue
            left_cp = control_points[idx]
            right_cp = control_points[idx + 1]
            #control points 5 and 3 and 2 are the left cp, 6 4 2 are the right cp
            left_cp_left_finger = np.array([left_cp[i] for i in [0,2, 4]])
            left_cp_right_finger = np.array([left_cp[i] for i in [0,3, 5]])
            right_cp_left_finger = np.array([right_cp[i] for i in [0,2, 4]])
            right_cp_right_finger = np.array([right_cp[i] for i in [0,3, 5]])
            left_cp_left_finger_center = left_cp_left_finger.mean(axis = 0)
            left_cp_right_finger_center = left_cp_right_finger.mean(axis = 0)
            right_cp_left_finger_center = right_cp_left_finger.mean(axis = 0)
            right_cp_right_finger_center = right_cp_right_finger.mean(axis = 0)
            gripper_vector_left = left_cp_right_finger_center - left_cp_left_finger_center
            gripper_vector_right = right_cp_right_finger_center - right_cp_left_finger_center
            gripper_lenght_left = np.linalg.norm(gripper_vector_left)
            gripper_lenght_right = np.linalg.norm(gripper_vector_right)
            grippper_left_widths.append(gripper_lenght_left)
            gripper_right_widths.append(gripper_lenght_right)
            gripper_left_vector_normalized = gripper_vector_left / gripper_lenght_left
            gripper_right_vector_normalized = gripper_vector_right / gripper_lenght_right
            projections_left = np.dot(numpy_og_pc - left_cp_left_finger_center, gripper_left_vector_normalized)
            projections_right = np.dot(numpy_og_pc - right_cp_left_finger_center, gripper_right_vector_normalized)
            is_between_left = np.logical_and(projections_left > 0, projections_left < gripper_lenght_left)
            is_between_right = np.logical_and(projections_right > 0, projections_right < gripper_lenght_right)

            def point_to_line_dist(points, line_start, line_end):
                line_start_to_point = points - line_start
                line_vector = line_end - line_start
                
                t = np.matmul(line_start_to_point, line_vector) / np.dot(line_vector, line_vector)
                t = np.clip(t, 0, 1)
                projection = line_start + t[:, None] * line_vector
                return np.linalg.norm(points - projection, axis = 1)
            distances_left = point_to_line_dist(numpy_og_pc, left_cp_left_finger_center, left_cp_right_finger_center)
            distances_right = point_to_line_dist(numpy_og_pc, right_cp_left_finger_center, right_cp_right_finger_center)
            is_within_left = distances_left < threshold_distance
            is_within_right = distances_right < threshold_distance
            is_within_left = is_within_left & is_between_left
            is_within_right = is_within_right & is_between_right
            #check if one of the points is within the gripper

            # check if the point does not collide with the object
            left_cp_base = np.array([left_cp[i] for i in [0, 1]])
            right_cp_base = np.array([right_cp[i] for i in [0, 1]])
            left_cp_i = np.expand_dims(left_cp, 1)
            right_cp_i = np.expand_dims(right_cp, 1)
            pc_expanded = np.expand_dims(numpy_og_pc, 0)
            distances_left = np.linalg.norm(left_cp_i - pc_expanded, axis = 2)
            distances_right = np.linalg.norm(right_cp_i - pc_expanded, axis = 2)
            min_distance_left = np.min(distances_left, axis = 0)
            min_distance_right = np.min(distances_right, axis = 0)
            is_collision_left = (min_distance_left < collision_threshold).any().item()
            is_collision_right = (min_distance_right < collision_threshold).any().item()

            if np.any(is_within_left) and np.any(is_within_right) and is_collision_left == False and is_collision_right == False:
                succesfull_grasps.append(grasps[idx])
                succesfull_grasps.append(grasps[idx + 1])
                succesfull_prob.append(succes_prob[int((idx+1)/2)])
            else:
                unsuccesfull_grasps.append(grasps[idx])
                unsuccesfull_grasps.append(grasps[idx + 1])
                # print("Found a succesfull grasp")
            #Skip the right grasp as they are paired




        # grasps = utils.control_points_from_grasps(control_points, 'tf', pc = None)
        control_points = utils.control_points_from_grasps(succesfull_grasps, 'cp', pc = og_pc)
        control_points = utils.control_points_from_grasps(unsuccesfull_grasps, 'cp', pc = og_pc)
        print("ammount of succesfull grasps after", len(succesfull_grasps))
        grasps = succesfull_grasps
        succes_prob = succesfull_prob
    
        return grasps, succes_prob